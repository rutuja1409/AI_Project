import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
from torchvision import transforms
import torch.nn.functional as F
from torch import nn


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


class NormalizeImage(object):
    def __call__(self, img):
        mean = torch.mean(img)
        std = torch.std(img)
        return (img - mean) / std


class Network(BaseNetwork):
    def __init__(
        self,
        unet,
        beta_schedule,
        module_name="sr3",
        sampling_timesteps=100,
        ddim_sampling_eta=0,
        **kwargs
    ):
        super(Network, self).__init__(**kwargs)
        if module_name == "sr3":
            from .sr3_modules.unet import UNet
        elif module_name == "guided_diffusion":
            from .guided_diffusion_modules.unet import UNet

        self.ddim_sampling_eta = ddim_sampling_eta
        self.sampling_timesteps = sampling_timesteps
        self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule
        self.transform = transforms.Compose([NormalizeImage()])
        self.bce_loss = nn.BCELoss()

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device("cuda"), phase="train"):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = (
            betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        )
        alphas = 1.0 - betas

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps

        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1.0, gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("gammas", to_torch(gammas))
        self.register_buffer("sqrt_recip_gammas", to_torch(np.sqrt(1.0 / gammas)))
        self.register_buffer("sqrt_recipm1_gammas", to_torch(np.sqrt(1.0 / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - gammas_prev) / (1.0 - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(gammas_prev) / (1.0 - gammas)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch((1.0 - gammas_prev) * np.sqrt(alphas) / (1.0 - gammas)),
        )

    def predict_noise_from_start(self, y_t, t, y0):
        return (extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t - y0) / extract(
            self.sqrt_recipm1_gammas, t, y_t.shape
        )

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t
            - extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat
            + extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, y_t.shape
        )
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level)
        y_0_hat = normalize_to_neg_one_to_one(F.tanh(y_0_hat))

        # normalize the image samples and binarize the image here
        y_0_hat[y_0_hat > 0.0] = 1.0
        y_0_hat[y_0_hat < 0.0] = -1.0

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t
        )
        return model_mean, posterior_log_variance, y_0_hat

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return sample_gammas.sqrt() * y_0 + (1 - sample_gammas).sqrt() * noise

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance, y_0_hat = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond
        )
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp(), y_0_hat

    @torch.no_grad()
    def ddim_sample(self, y_cond, return_all_timesteps=True):
        shape = y_cond.shape
        batch, device, total_timesteps, sampling_timesteps, eta = (
            y_cond.shape[0],
            y_cond.get_device(),
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn((shape[0], 1, shape[2], shape[3]), device=device)
        imgs = [img]

        y_0_hat = None
        ret_arr = img

        i = 0
        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            t = torch.full((batch,), time, device=device, dtype=torch.long)
            noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(img.device)
            y_0_hat = self.denoise_fn(torch.cat([y_cond, img], dim=1), noise_level)
            y_0_hat = normalize_to_neg_one_to_one(F.tanh(y_0_hat))

            # normalize the image samples and binarize the image here
            y_0_hat[y_0_hat > 0.0] = 1.0
            y_0_hat[y_0_hat < 0.0] = -1.0

            pred_noise = self.predict_noise_from_start(img, t, y_0_hat)

            if time_next < 0:
                img = y_0_hat
                imgs.append(img)
                continue

            alpha = self.gammas[time]
            alpha_next = self.gammas[time_next]

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = y_0_hat * alpha_next.sqrt() + c * pred_noise + sigma * noise

            if i % 50 == 0:
                ret_arr = torch.cat([ret_arr, img], dim=0)
            i += 1

        return img, ret_arr

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
        # if self.is_ddim_sampling:
        #     y_cond = normalize_to_neg_one_to_one(y_cond)
        #     return self.ddim_sample(y_cond)

        b, *_ = y_cond.shape

        assert (
            self.num_timesteps > sample_num
        ), "num_timesteps must greater than sample_num"
        sample_inter = self.num_timesteps // sample_num

        b, c, h, w = y_cond.shape
        y_t = default(y_t, lambda: torch.randn(b, 1, h, w, device=y_cond.device))
        ret_arr = y_t
        y_cond = normalize_to_neg_one_to_one(y_cond)
        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t, y_0_hat = self.p_sample(y_t, t, y_cond=y_cond)
            if mask is not None:
                y_t = y_0 * (1.0 - mask) + mask * y_t
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
                ret_arr = torch.cat([ret_arr, y_0_hat], dim=0)
        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, noise=None):
        # sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        print("t", t)
        gamma_t1 = extract(self.gammas, t - 1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2 - gamma_t1) * torch.rand(
            (b, 1), device=y_0.device
        ) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise
        )

        # print(
        #     "y_0",
        #     y_0.min(),
        #     y_0.max(),
        #     y_0.mean(),
        #     y_0.std(),
        # )
        # print("y_cond", y_cond.min(), y_cond.max(), y_cond.mean(), y_cond.std())
        # print("y_noisy", y_noisy.min(), y_noisy.max(), y_noisy.mean(), y_noisy.std())
        y_0_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
        # print("y_0_hat", y_0_hat.min(), y_0_hat.max(), y_0_hat.mean(), y_0_hat.std())
        loss = self.loss_fn(F.tanh(y_0_hat), y_0)
        # print(
        #     "F.tanh(y_0_hat)",
        #     F.tanh(y_0_hat).min(),
        #     F.tanh(y_0_hat).max(),
        #     F.tanh(y_0_hat).mean(),
        #     F.tanh(y_0_hat).std(),
        #     print("y_0", y_0.shape, y_0_hat.shape),
        # )

        print("y_noisy", y_noisy.shape)

        import matplotlib.pyplot as plt

        plt.imshow(y_noisy[0].permute(1, 2, 0).detach().cpu(), cmap="gray")
        plt.show()

        plt.imshow(F.tanh(y_0_hat[0]).permute(1, 2, 0).detach().cpu(), cmap="gray")
        plt.show()

        plt.imshow(y_0[0].permute(1, 2, 0).detach().cpu(), cmap="gray")
        plt.show()
        return loss


# gaussian diffusion trainer class
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape=(1, 1, 1, 1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64
    )
    return betas


def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3
):
    if schedule == "quad":
        betas = (
            np.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64
            )
            ** 2
        )
    elif schedule == "linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "warmup10":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == "warmup50":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == "const":
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas
