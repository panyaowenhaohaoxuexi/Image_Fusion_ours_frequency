# -*- coding: utf-8 -*-
"""网络统一入口。"""
from net.encoder.simple_encoder import SimpleSharedEncoder
from net.fusion.base_fusion import SimpleBaseFusion
from net.decoder.simple_decoder import SimpleDecoder


class SharedEncoder(SimpleSharedEncoder):
    pass


class BaseFusion(SimpleBaseFusion):
    pass


class FusionDecoder(SimpleDecoder):
    pass


# 兼容旧命名
class Restormer_Encoder(SharedEncoder):
    pass


class BaseFeatureExtraction(BaseFusion):
    pass


class Restormer_Decoder(FusionDecoder):
    pass
