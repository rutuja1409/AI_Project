from pathlib import Path
import pickle
import random
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import numpy as np

import torchvision.transforms.functional as F
import cv2
import numpy as np
import random
import torchvision.transforms.functional as TF


class SquarePad:
    """
    Pads the image to right side with given backgroud pixel values
    """

    pad_value: int = 255

    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int(max_wh - w)
        vp = int(max_wh - h)
        padding = (0, 0, hp, vp)
        return F.pad(image, padding, self.pad_value, "constant")


class ShabbyPages(data.Dataset):
    def __init__(self, dataset_path, image_size, stage="train", use_gray_gt=True):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.stage = stage
        self.use_gray_gt = use_gray_gt
        self.stage = "validation" if self.stage == "val" else self.stage
        if self.use_gray_gt:
            mean = [0.5]
            std = [0.5]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.pad = transforms.Compose([SquarePad()])
        if self.stage == "train":
            self.tfs = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

            self.gray_tfs = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            self.tfs = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

            self.gray_tfs = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

        from datadings.reader import MsgpackReader

        self.data_reader = MsgpackReader(
            Path(self.dataset_path) / "PNG" / self.stage / "train_512x512.msgpack"
        )

    def __getitem__(self, index):
        import io

        sample = pickle.loads(self.data_reader[index]["data"])
        cond_image = Image.open(io.BytesIO(sample["image"]))
        gt_image = Image.open(io.BytesIO(sample["gt_image"])).convert("L")
        cond_image = self.pad(cond_image)
        gt_image = self.pad(gt_image)

        # apply data augmentation
        if self.stage == "train":
            # random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                gt_image, output_size=(self.image_size, self.image_size)
            )
            i, j, h, w = transforms.RandomCrop.get_params(
                cond_image, output_size=(self.image_size, self.image_size)
            )
            cond_image = TF.crop(cond_image, i, j, h, w)
            gt_image = TF.crop(gt_image, i, j, h, w)

            # random horizontal flipping
            if random.random() > 0.5:
                gt_image = TF.hflip(gt_image)
                cond_image = TF.hflip(cond_image)

            # random vertical flipping
            if random.random() > 0.5:
                gt_image = TF.vflip(gt_image)
                cond_image = TF.vflip(cond_image)

        gt_image = np.array(gt_image.convert("L"))
        ret, gt_image = cv2.threshold(
            gt_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if gt_image.mean() == 0:
            gt_image = 255 - gt_image
        gt_image = Image.fromarray(gt_image)

        # binarize gt_images
        cond_image = self.tfs(cond_image)
        gt_image = self.gray_tfs(gt_image)

        # import matplotlib.pyplot as plt

        # print("stage", self.stage)
        # print(cond_image.shape, gt_image.shape)

        # plt.imshow(cond_image.permute(1, 2, 0))
        # plt.show()

        # plt.imshow(gt_image.permute(1, 2, 0), cmap="gray", vmin=-1, vmax=1)
        # plt.show()
        # print(
        #     "cond_image",
        #     cond_image.min(),
        #     cond_image.max(),
        #     cond_image.mean(),
        #     cond_image.std(),
        # )
        # print(
        #     "gt_image", gt_image.min(), gt_image.max(), gt_image.mean(), gt_image.std()
        # )
        ret = {}
        ret["gt_image"] = gt_image
        ret["cond_image"] = cond_image
        ret["path"] = sample["image_file_path"]
        return ret

    def __len__(self):
        return len(self.data_reader)


# d = ShabbyPages(
#     **{
#         "dataset_path": "/run/user/3841/gvfs/sftp:host=login1.pegasus.kl.dfki.de/ds/documents/ShabbyPages",
#         "image_size": 256,
#         "stage": "train",
#         "use_gray_gt": True,
#     }
# )
# for i in range(10):
#     d[i]
