# -*- coding: utf-8 -*-
"""Unified network entrypoints."""
from net.encoder.simple_encoder import SimpleSharedEncoder
from net.fusion.base_fusion import SimpleBaseFusion
from net.fusion.text_conditioned_spatial_fusion import TGCSF as _TGCSF
from net.fusion.spatial_compensation import SpatialResidualCompensation as _SpatialResidualCompensation
from net.decoder.simple_decoder import SimpleDecoder
from net.intent import DualStreamIntentMLP
from net.bfsc import BFSC


class SharedEncoder(SimpleSharedEncoder):
    pass


class BaseFusion(SimpleBaseFusion):
    pass


class TGCSF(_TGCSF):
    pass


class TextConditionedSpatialFusion(TGCSF):
    pass


class FusionDecoder(SimpleDecoder):
    pass


class SpatialResidualCompensation(_SpatialResidualCompensation):
    pass


class Restormer_Encoder(SharedEncoder):
    pass


class BaseFeatureExtraction(BaseFusion):
    pass


class Restormer_Decoder(FusionDecoder):
    pass
