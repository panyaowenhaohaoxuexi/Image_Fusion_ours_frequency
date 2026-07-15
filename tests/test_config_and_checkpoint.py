import importlib
import unittest

import torch.nn as nn


class ConfigAndCheckpointTests(unittest.TestCase):
    def test_config_derives_checkpoint_tag_from_query_variant(self):
        config = importlib.import_module("config")

        self.assertTrue(config.USE_CLIP_IMAGE_QUERY)
        self.assertEqual(config.MODEL_VERSION, "v13_components")
        self.assertEqual(config.QUERY_VARIANT, "clip_image_query")
        self.assertEqual(config.CHECKPOINT_TAG, "v13_components_clip_image_query")

    def test_checkpoint_metadata_rejects_query_variant_mismatch(self):
        inference = importlib.import_module("test")
        config = importlib.import_module("config")
        intent_generator = nn.Identity()
        checkpoint = {
            "model_version": config.MODEL_VERSION,
            "use_clip_image_query": False,
            "intent_generator_type": "Identity",
        }

        with self.assertRaisesRegex(RuntimeError, "query variant mismatch"):
            inference.validate_checkpoint_metadata(checkpoint, intent_generator)

    def test_network_exports_both_intent_variants(self):
        network = importlib.import_module("net.Network")

        self.assertTrue(hasattr(network, "DualDomainTextIntentGenerator"))
        self.assertTrue(hasattr(network, "DualStreamIntentMLP"))

    def test_train_and_test_share_decoder_config_constants(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for filename in ("train.py", "test.py"):
            source = (root / filename).read_text(encoding="utf-8")
            decoder_constructor = source.split("FusionDecoder(", 1)[1].split(")).to(device)", 1)[0]
            self.assertIn("inner_dim=DECODER_INNER_DIM", decoder_constructor)
            self.assertIn("num_blocks=DECODER_NUM_BLOCKS", decoder_constructor)
            self.assertIn("max_residual_scale=DECODER_MAX_RESIDUAL_SCALE", decoder_constructor)
            self.assertNotIn("inner_dim=24", decoder_constructor)
            self.assertNotIn("num_blocks=1", decoder_constructor)
