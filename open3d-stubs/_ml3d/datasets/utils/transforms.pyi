"""
This type stub file was generated by pyright.
"""

from .operations import *

def trans_normalize(pc, feat, t_normalize): ...
def trans_augment(points, t_augment):
    """Implementation of an augmentation transform for point clouds."""
    ...

def trans_crop_pc(points, feat, labels, search_tree, pick_idx, num_points): ...
def in_range_bev(box_range, box): ...

class ObjdetAugmentation:
    """Class consisting different augmentation for Object Detection."""

    @staticmethod
    def PointShuffle(data): ...
    @staticmethod
    def ObjectRangeFilter(data, pcd_range): ...
    @staticmethod
    def ObjectSample(data, db_boxes_dict, sample_dict): ...
    @staticmethod
    def ObjectNoise(input, trans_std=..., rot_range=..., num_try=...): ...
