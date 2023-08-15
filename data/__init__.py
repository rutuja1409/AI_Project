from functools import partial
import numpy as np
import random

from torch.utils.data.distributed import DistributedSampler
from torch import Generator, randperm
from torch.utils.data import DataLoader, Subset

import core.util as Util
from core.praser import init_obj


def define_dataloader(logger, opt):
    """ create train/test dataloader and validation dataloader,  validation dataloader is None when phase is test or not GPU 0 """
    '''create dataset and set random seed'''
    dataloader_args = opt['datasets'][opt['phase']]['dataloader']['args']
    worker_init_fn = partial(Util.set_seed, gl_seed=opt['seed'])

    phase_dataset, val_dataset = define_dataset(logger, opt)

    '''create datasampler'''
    data_sampler = None
    if opt['distributed']:
        data_sampler = DistributedSampler(phase_dataset, shuffle=dataloader_args.get('shuffle', False), num_replicas=opt['world_size'], rank=opt['global_rank'])
        dataloader_args.update({'shuffle':False}) # sampler option is mutually exclusive with shuffle 
    

    ''' create dataloader and validation dataloader '''
    dataloader = DataLoader(phase_dataset, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)
    ''' val_dataloader don't use DistributedSampler to run only GPU 0! '''
    if opt['global_rank']==0 and val_dataset is not None:
        val_dataloader_args = opt['datasets']['validation']['dataloader']['args']  # Use val_args from validation section
        val_dataloader = DataLoader(val_dataset, worker_init_fn=worker_init_fn, **val_dataloader_args)
    else:
        val_dataloader = None
    return dataloader, val_dataloader


def define_dataset(logger, opt):
    ''' loading Dataset() class from given file's name '''
    dataset_opt = opt['datasets'][opt['phase']]['which_dataset']
    phase_dataset = init_obj(dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset')
    
    if opt['phase'] == 'train':
        val_dataset_opt = opt['datasets']['validation']['which_dataset']
        val_dataset = init_obj(val_dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset')

    logger.info('Dataset for {} have {} samples.'.format(opt['phase'], len(phase_dataset)))
    val_list = random.sample(range(1, len(val_dataset)), 100)
    if opt['phase'] == 'train' and val_dataset is not None:
        logger.info('Dataset for {} have {} samples.'.format('val', len(val_list)))
    
    val_dataset = Subset(val_dataset, val_list)
        
    return phase_dataset, val_dataset