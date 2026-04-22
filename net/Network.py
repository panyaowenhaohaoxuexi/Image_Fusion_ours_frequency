# -*- coding: utf-8 -*-
"""兼容入口：内部实现已经替换为干净版本。"""
from net.encoder.simple_encoder import SimpleSharedEncoder
from net.fusion.base_fusion import SimpleBaseFusion
from net.decoder.simple_decoder import SimpleDecoder

class Restormer_Encoder(SimpleSharedEncoder):
    pass

class BaseFeatureExtraction(SimpleBaseFusion):
    pass

class Restormer_Decoder(SimpleDecoder):
    pass
