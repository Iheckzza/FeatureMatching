import os
import math
import random
import sys
from collections import defaultdict, abc
import pprint
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from os import path as osp
from joblib import Parallel, delayed

import torch.functional
from torch.utils.data.dataset import Dataset

import torch
import numpy as np
import pytorch_lightning as pl
from torch import distributed as dist
from matplotlib import pyplot as plt
from torch.utils.data import Sampler, Dataset, DataLoader, ConcatDataset, DistributedSampler, RandomSampler, dataloader, \
    random_split

from .network.net import net
from Loftr.src.loftr import LoFTR
# from .network.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from .network.utils.supervision_new import compute_supervision_coarse, compute_supervision_fine
from .datasets.data_preprocessing import data_preprocess
from .losses.loss import Loss
from .optimizers import build_optimizer, build_scheduler
from .utils.metrics import compute_symmetrical_epipolar_errors, compute_pose_errors_new, aggregate_metrics, \
    compute_pose_errors
from .utils.plotting import make_matching_figures
from .utils.comm import gather, all_gather
from .utils.misc import lower_config, flattenList, tqdm_joblib
from .utils.profiler import PassThroughProfiler
from .utils.RandomSampler import RandomConcatSampler
from .datasets.scared_new2 import ScaredDataset
from .datasets.endoslam import EndoDataset
from .datasets.unity_data import UnityDataset


class MultiSceneDataModule(pl.LightningDataModule):
    def __init__(self, args, config):
        super().__init__()

        self.data_enhance = config.DATASET.DATA_ENHANCE
        self.data_type = config.DATASET.DATA_TYPE
        self.read_img_gray = config.DATASET.IMG_READ_GRAY
        self.debug_flag = config.DATASET.DEBUG

        # training and validating
        self.trainval_data_root = config.DATASET.TRAINVAL_DATA_ROOT

        # testing
        self.test_data_root = config.DATASET.TEST_DATA_ROOT

        # loader parameters
        self.train_loader_params = {'batch_size': args.batch_size,
                                    'num_workers': args.num_workers,
                                    'pin_memory': getattr(args, 'pin_memory', False)}
        self.val_loader_params = {'batch_size': 1,
                                  'shuffle': False,
                                  'num_workers': args.num_workers,
                                  'pin_memory': getattr(args, 'pin_memory', False)}
        self.test_loader_params = {'batch_size': 1,
                                   'shuffle': False,
                                   'num_workers': args.num_workers,
                                   'pin_memory': False}

        # sampler
        self.data_sampler = config.TRAINER.DATA_SAMPLER
        self.n_samples_per_subset = config.TRAINER.N_SAMPLES_PER_SUBSET
        self.subset_replacement = config.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT
        self.shuffle = config.TRAINER.SB_SUBSET_SHUFFLE
        self.repeat = config.TRAINER.SB_REPEAT

        self.lighting_data = config.DATASET.LIGHTING_DATA

        self.seed = config.TRAINER.SEED

    def setup(self, stage=None):
        assert stage in ['fit', 'test'], "stage must be either fit or test"
        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        except AssertionError as ae:
            self.world_size = 1
            self.rank = 0
            logger.warning(str(ae) + " (set wolrd_size=1 and rank=0)")

        if stage == 'fit':
            self.train_dataset, self.val_dataset = self._setup_dataset(
                data_root=self.trainval_data_root,
                mode='trainval',
                data_enhance=self.data_enhance,
                data_type=self.data_type)
            logger.info(f' Train/Val Dataset loaded!')

        elif stage == 'test':
            self.test_dataset = self._setup_dataset(
                data_root=self.test_data_root,
                mode='test',
                data_enhance=1,
                data_type=self.data_type)
            logger.info(f' Test Dataset loaded!')
        else:
            raise NotImplementedError()

    def _setup_dataset(self, data_root, mode, data_enhance, data_type):
        datasets_train, datasets_val, datasets_test = [], [], []

        if data_type == 'Scared':
            dataset = ScaredDataset
        elif data_type == 'EndoSlam':
            dataset = EndoDataset
        elif data_type == 'Unity':
            dataset = UnityDataset
        else:
            raise IOError('Wrong dataset type!')

        if mode == 'trainval':
            if data_type != 'Unity':
                datasets_total = []
                for ii in sorted(os.listdir(data_root), key=lambda x: x.split()[-1]):
                    data_name_list = os.path.join(data_root, ii)
                    for jj in sorted(os.listdir(data_name_list), key=lambda x: x.split()[-1]):
                        datasets_total.append(
                            dataset(os.path.join(data_name_list, jj), mode=mode, data_enhance=data_enhance,
                                    read_img_gray=self.read_img_gray, lighting_data=self.lighting_data,
                                    debug_flag=self.debug_flag))

                train_num = round(len(datasets_total) * 0.85)
                train_data = random.sample(datasets_total, train_num)
                val_data = [i for i in datasets_total if i not in train_data]

                return ConcatDataset(train_data), ConcatDataset(val_data)
            else:
                datasets_total = []
                for ii in sorted(os.listdir(data_root), key=lambda x: x.split()[-1]):
                    data_name_list = os.path.join(data_root, ii)
                    datasets_total.append(dataset(data_name_list, mode=mode, data_enhance=data_enhance,
                                                  read_img_gray=self.read_img_gray, lighting_data=self.lighting_data,
                                                  debug_flag=self.debug_flag))

                train_data = datasets_total[:2]
                val_data = datasets_total[2:]
                return ConcatDataset(train_data), ConcatDataset(val_data)
        # else:
        #     for ii in sorted(os.listdir(data_root), key=lambda x: x.split()[-1]):
        #         data_name_list = os.path.join(data_root, ii)
        #         for jj in sorted(os.listdir(data_name_list), key=lambda x: x.split()[-1]):
        #             datasets_test.append(
        #                 dataset(os.path.join(data_name_list, jj), mode=mode, data_enhance=data_enhance,
        #                         read_img_gray=self.read_img_gray, lighting_data=self.lighting_data,
        #                         debug_flag=self.debug_flag))
        #     return ConcatDataset(datasets_test)

    def train_dataloader(self):
        logger.info(
            f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).')
        if self.data_sampler == 'scene_balance':
            sampler = RandomConcatSampler(self.train_dataset,
                                          self.n_samples_per_subset,
                                          self.subset_replacement,
                                          self.shuffle,
                                          self.repeat,
                                          self.seed)
        else:
            sampler = DistributedSampler(self.train_dataset, shuffle=True)

        train_dataloader = DataLoader(self.train_dataset, sampler=sampler, **self.train_loader_params)
        return train_dataloader

    def val_dataloader(self):
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')

        sampler = DistributedSampler(self.val_dataset, shuffle=False)
        return DataLoader(self.val_dataset, sampler=sampler, **self.val_loader_params)

    # def test_dataloader(self, *args, **kwargs):
    #     sampler = DistributedSampler(self.test_dataset, shuffle=False)
    #     return DataLoader(self.test_dataset, sampler=sampler, **self.test_loader_params)


class Lightning(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None, loftr_cfg=None, loftr_ckpt=None):
        super().__init__()
        self.config = config
        _config = lower_config(self.config)
        self.module_cfg = lower_config(_config['module'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher
        self.matcher = net(config=_config['module'])
        self.loss = Loss(_config)

        if loftr_cfg:
            self.loftr = LoFTR(config=loftr_cfg)
        if loftr_ckpt:
            state_dict_loftr = torch.load(loftr_ckpt, map_location='cpu')['state_dict']
            self.loftr.load_state_dict(state_dict_loftr, strict=False)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.matcher.load_state_dict(state_dict, strict=False)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")

        # Testing
        self.dump_dir = dump_dir

    def _trainval_inference(self, batch):
        with self.profiler.profile('Data preprocessing'):
            data_preprocess(batch)

        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("Module"):
            self.matcher(batch)

        with self.profiler.profile("Compute fine supervision"):
            compute_supervision_fine(batch)

        with self.profiler.profile("Compute losses"):
            self.loss(batch)

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.logger.experiment.add_scalar(f'train/{k}', v, self.global_step)

        print("current train loss: " + str(batch['loss_scalars']))
        return {'loss': batch['loss']}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar('train/avg_loss_on_epoch', avg_loss, global_step=self.current_epoch)

    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            compute_pose_errors(batch)
            # compute_pose_errors_new(batch)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['image0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs']}
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names

    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        ret_dict, _ = self._compute_metrics(batch)

        print('valloss' + str(batch['loss_scalars']))

        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars']}

    def validation_epoch_end(self, outputs):
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)

        loss_final_mean = 0.
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpuG
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            for thr in [5, 10, 20]:
                multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])

            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)
                    loss_final_mean += mean_v

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)

        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))  # ckpt monitors on this

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp,
                       using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                     (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                     abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
