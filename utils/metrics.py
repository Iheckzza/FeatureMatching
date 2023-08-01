import torch
import cv2
import numpy as np
from collections import OrderedDict
from loguru import logger
from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous


# --- METRICS ---

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    # t_gt = T_0to1[:3, 3]
    # n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    # t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    # t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    # if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
    #     t_err = 0

    t_err = np.linalg.norm((T_0to1[:3, 3] - t), ord=2)

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


#######重投影误差估计##########
def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """

    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    pts0 = pts0.to(torch.float32)
    Ep0 = pts0 @ E.T  # [N, 3]
    pts1 = pts1.to(torch.float32)
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    d = p1Ep0 ** 2 * (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2) + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2))  # N
    return d


def compute_symmetrical_epipolar_errors(data):
    """ 
    Update:
        data (dict):{"epi_errs": [M]}
    """
    Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
    E_mat = Tx @ data['T_0to1'][:, :3, :3]

    m_bids = data['m_bids']
    pts0 = data['mkpts0_f'][:, :2]
    pts1 = data['mkpts1_f'][:, :2]

    epi_errs = []
    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        epi_errs.append(
            symmetric_epipolar_distance(pts0[mask], pts1[mask], E_mat[bs], data['K0'][bs], data['K1'][bs]))

    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({'epi_errs': epi_errs})


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    # normalize keypoints
    kpts0_norm = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1_norm = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    focal_length = 0.5 * (K0[0, 0] + K0[1, 1])
    principle_point = (K0[0, 2], K0[1, 2])
    E, mask = cv2.findEssentialMat(kpts0, kpts1, focal_length, principle_point, cv2.RANSAC, 0.999, 1.0)

    # # normalize ransac threshold
    # ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    # E_norm, mask_norm = cv2.findEssentialMat(
    #     kpts0_norm, kpts1_norm, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)

    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


def compute_pose_errors_new(data):
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})

    T_0to1_pred = data['T_0to1_pred'].cpu().numpy()
    T_0to1 = data['T_0to1'].cpu().numpy()

    R, t = T_0to1_pred[0, :3, :3], T_0to1_pred[0, :3, 3:].flatten()
    t_err, R_err = relative_pose_error(T_0to1[0], R, t, ignore_gt_t_thr=0.0)
    data['R_errs'].append(R_err)
    data['t_errs'].append(t_err)


def compute_pose_errors(data):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    # pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    # conf = config.TRAINER.RANSAC_CONF  # 0.99999
    pixel_thr = 0.5
    conf = 0.99999
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})

    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'][:, :2].cpu().numpy()
    pts1 = data['mkpts1_f'][:, :2].cpu().numpy()
    K0 = data['K0'].cpu().numpy()
    K1 = data['K1'].cpu().numpy()
    T_0to1 = data['T_0to1'].cpu().numpy()

    for bs in range(K0.shape[0]):
        mask = m_bids == bs
        ret = estimate_pose(pts0[mask], pts1[mask], K0[bs], K1[bs], pixel_thr, conf=conf)

        if ret is None:
            data['R_errs'].append(np.inf)
            data['t_errs'].append(np.inf)
            data['inliers'].append(np.array([]).astype(np.bool))
        else:
            # R, t = data['T_0to1_pred'][:, :3, :3], data['T_0to1_pred'][:, :3, 3:]
            R, t, _ = ret
            t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
            data['R_errs'].append(R_err)
            data['t_errs'].append(t_err)


def error_auc(errors, thresholds, ret_dict=True):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    if ret_dict:
        return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}
    else:
        return aucs


def epidist_prec(errors, thresholds, ret_dict=True):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs


def aggregate_metrics(metrics, epi_err_thr=5e-4):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # pose auc
    angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
    aucs = error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

    # matching precision
    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(np.array(metrics['epi_errs'], dtype=object)[unq_ids], dist_thresholds, True)  # (prec@err_thr)

    return {**aucs, **precs}
