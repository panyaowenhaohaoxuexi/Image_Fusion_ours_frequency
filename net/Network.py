# -*- coding: utf-8 -*-
"""Unified network entrypoints."""
from net.encoder.simple_encoder import SimpleSharedEncoder
from net.fusion.text_conditioned_spatial_fusion import TGCSF as _TGCSF
from net.decoder.simple_decoder import SimpleDecoder
from net.intent import DualDomainTextIntentGenerator
from net.dda import DDA


class SharedEncoder(SimpleSharedEncoder):
    pass


class TGCSF(_TGCSF):
    pass


class TextConditionedSpatialFusion(TGCSF):
    pass


class FusionDecoder(SimpleDecoder):
    pass


class Restormer_Encoder(SharedEncoder):
    pass


class Restormer_Decoder(FusionDecoder):
    pass
