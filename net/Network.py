# -*- coding: utf-8 -*-
"""网络统一入口。"""
from net.encoder.simple_encoder import SimpleSharedEncoder
from net.fusion.base_fusion import SimpleBaseFusion
from net.fusion.spatial_compensation import SpatialResidualCompensation as _SpatialResidualCompensation
from net.decoder.simple_decoder import SimpleDecoder


class SharedEncoder(SimpleSharedEncoder):
    """当前项目使用的共享编码器（Restormer 轻量版）。"""
    pass


class BaseFusion(SimpleBaseFusion):
    pass


class FusionDecoder(SimpleDecoder):
    """当前项目使用的解码器（Restormer 轻量版）。"""
    pass


class SpatialResidualCompensation(_SpatialResidualCompensation):
    """频率参照引导的空间残差补偿模块。"""
    pass


# 兼容旧命名
class Restormer_Encoder(SharedEncoder):
    pass


class BaseFeatureExtraction(BaseFusion):
    pass


class Restormer_Decoder(FusionDecoder):
    pass