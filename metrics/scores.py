import numpy as np


def dice_coeff(gt_seg, est_seg):
    """

    :param gt_seg: the ground truth segment to compare with
    :param est_seg: the estimated segment to evaluate.
    :return: score between 0 and 1, where 1 is the highest score.
    """
    est_mapping, gt_mapping =get_binary_map_for_segments(est_seg, gt_seg)
    sum_of_volumes = (np.sum(est_mapping) + np.sum(gt_mapping))
    return np.sum(gt_mapping[est_mapping == 1]) * 2.0 / sum_of_volumes


def vod_score(gt_seg, est_seg):
    est_mapping, gt_mapping = get_binary_map_for_segments(est_seg, gt_seg)
    ct_union = est_mapping + gt_mapping
    ct_union[ct_union > 0] = 1
    sum_of_volumes = np.sum(ct_union)
    if sum_of_volumes > 0:
        return 1 - np.sum(gt_mapping[est_mapping == 1]) / sum_of_volumes
    return 0


def get_binary_map_for_segments(est_seg, gt_seg):
    gt_mapping, est_mapping = np.zeros(gt_seg.shape), np.zeros(gt_seg.shape)
    gt_mapping[gt_seg != 0], est_mapping[est_seg != 0] = 1, 1
    return est_mapping, gt_mapping
