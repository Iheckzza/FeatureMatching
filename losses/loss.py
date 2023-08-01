from loguru import logger
import numpy as np

from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['module']['loss']
        self.match_type = 'dual_softmax'
        self.sparse_spvs = self.config['module']['match_coarse']['sparse_spvs']

        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']

        # pose-loss
        self.pose_loss_cal_flag = self.loss_config['pose_loss_cal_flag']
        self.quot_loss = nn.MSELoss(reduction='sum')
        self.translate_loss = nn.MSELoss(reduction='mean')

    def compute_coarse_loss(self, conf, conf_gt):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w

        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'cross_entropy':
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            loss_pos = - torch.log(conf[pos_mask])
            loss_neg = - torch.log(1 - conf[neg_mask])

            coarse_loss = c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
            return coarse_loss

        elif self.loss_config['coarse_type'] == 'focal':
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']

            if self.sparse_spvs:
                pos_conf = conf[pos_mask]
                loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()

                loss = c_pos_w * loss_pos.mean()
                return loss

            else:
                loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
                loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed

    def compute_fine_loss(self, expec_f_0, expec_f_1, expec_f_gt_0, expec_f_gt_1):

        if expec_f_0.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                return 0.
            else:
                return None

        std_0 = expec_f_0[:, 2]
        std_1 = expec_f_1[:, 2]
        inverse_std_0 = 1. / torch.clamp(std_0, min=1e-10)
        inverse_std_1 = 1. / torch.clamp(std_1, min=1e-10)
        weight_0 = (inverse_std_0 / torch.mean(inverse_std_0)).detach()  # avoid minizing loss through increase std
        weight_1 = (inverse_std_1 / torch.mean(inverse_std_1)).detach()

        non_zero_gt_0 = torch.where(expec_f_gt_0[:, 0] != 0)
        non_zero_gt_1 = torch.where(expec_f_gt_1[:, 0] != 0)

        offset_l2_0 = ((((expec_f_0[:, :2][non_zero_gt_0] - expec_f_gt_0[non_zero_gt_0])) / 7) ** 2).sum(-1)
        offset_l2_1 = ((((expec_f_1[:, :2][non_zero_gt_1] - expec_f_gt_1[non_zero_gt_1])) / 7) ** 2).sum(-1)

        # l2 loss with std
        # offset_l2_0 = ((((expec_f_0[:, :2] - expec_f_gt_0)) / 7) ** 2).sum(-1)
        # offset_l2_1 = ((((expec_f_1[:, :2] - expec_f_gt_1)) / 7) ** 2).sum(-1)

        loss = (offset_l2_0 * weight_0[non_zero_gt_0]).mean() + (offset_l2_1 * weight_1[non_zero_gt_1]).mean()

        return loss

    def relative_pose_error(self, T_0to1, R, t):

        T_0to1 = np.array(T_0to1[0].data.cpu())
        R = np.array(R[0].data.cpu())
        t = np.array(t[0].data.cpu())

        t_err = np.linalg.norm((T_0to1[:3, 3] - t), ord=2)

        # angle error between 2 rotation matrices
        R_gt = T_0to1[:3, :3]
        cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
        cos = np.clip(cos, -1., 1.)  # handle numercial errors
        R_err = np.rad2deg(np.abs(np.arccos(cos)))

        return R_err, t_err

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(data['conf_matrix'], data['conf_matrix_gt'])
        loss = loss_c * self.loss_config['coarse_weight']
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # 2. fine-level loss
        loss_f = self.compute_fine_loss(data['mkpts0_f'], data['mkpts1_f'], data['expec_f_gt_0'],
                                        data['expec_f_gt_1'])
        if loss_f is not None:
            loss += loss_f * self.loss_config['fine_weight']
            loss_scalars.update({"loss_f": loss_f.clone().detach().cpu()})
            if self.training is False:
                loss_scalars.update({'loss_f': torch.tensor(1.)})
        else:
            assert self.training is False
            loss_scalars.update({'loss_f': torch.tensor(1.)})

        # 3. pose loss
        if self.pose_loss_cal_flag == 'old':
            t_err, R_err = self.relative_pose_error(data['T_0to1'], data['T_0to1_pred'][:, :3, :3],
                                                    data['T_0to1_pred'][:, :3, 3:])
            t_inv_err, R_inv_err = self.relative_pose_error(data['T_1to0'], data['T_1to0_pred'][:, :3, :3],
                                                            data['T_1to0_pred'][:, :3, 3:])

            R_e = torch.tensor(np.mean([R_err, R_inv_err]), device=data['image0'].device)
            t_e = torch.tensor(np.mean([t_err, t_inv_err]), device=data['image0'].device)

            loss_pose = R_e * self.loss_config['r_weight'] + t_e * self.loss_config['t_weight']
            loss_pose = torch.log(loss_pose + 1)
        elif self.pose_loss_cal_flag == 'new':
            quto_loss = self.quot_loss(data['quot_0to1'].float(), data['quot_pred'][:, 0, :].float())
            quto_inv_loss = self.quot_loss(data['quot_1to0'].float(), data['quot_pred'][:, 1, :].float())
            t_loss = self.translate_loss(data['T_0to1'][0, :3, 3:].T.float(), data['translate_pred'][:, 0, :].float())
            t_inv_loss = self.translate_loss(data['T_1to0'][0, :3, 3:].T.float(),
                                             data['translate_pred'][:, 1, :].float())

            R_e = torch.mean(torch.concat([quto_loss[None], quto_inv_loss[None]]), dtype=torch.float32)
            t_e = torch.mean(torch.concat([t_loss[None], t_inv_loss[None]]), dtype=torch.float32)

            loss_pose = R_e * self.loss_config['r_weight'] + torch.log(t_e + 1) * self.loss_config['t_weight']
        else:
            loss_pose = torch.tensor(0., device=data['mkpts0_f'].device)

        # loss += loss_pose
        loss_scalars.update({"loss_pose": loss_pose.clone().detach().cpu()})
        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
