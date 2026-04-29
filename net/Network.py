# -*- coding: utf-8 -*-
"""网络统一入口。"""
from net.encoder.simple_encoder import SimpleSharedEncoder
from net.fusion.base_fusion import SimpleBaseFusion
from net.fusion.text_conditioned_spatial_fusion import TextConditionedSpatialAdaptiveFusion as _TextConditionedSpatialAdaptiveFusion
from net.fusion.spatial_compensation import SpatialResidualCompensation as _SpatialResidualCompensation
from net.decoder.simple_decoder import SimpleDecoder


class SharedEncoder(SimpleSharedEncoder):
    """当前项目使用的共享编码器（轻量 Restormer）。"""
    pass


class BaseFusion(SimpleBaseFusion):
    pass


class TextConditionedSpatialFusion(_TextConditionedSpatialAdaptiveFusion):
    """Text Intent-conditioned 空间自适应融合分支。"""
    pass


class FusionDecoder(SimpleDecoder):
    """当前项目使用的解码器（轻量 Restormer）。"""
    pass


class SpatialResidualCompensation(_SpatialResidualCompensation):
    """历史兼容：旧版频率参照空间残差补偿模块。"""
    pass


# 兼容旧命名
class Restormer_Encoder(SharedEncoder):
    pass


class BaseFeatureExtraction(BaseFusion):
    pass


class Restormer_Decoder(FusionDecoder):
    pass
