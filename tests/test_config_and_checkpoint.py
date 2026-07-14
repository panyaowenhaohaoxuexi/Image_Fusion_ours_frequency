import importlib
import unittest

import torch.nn as nn


class ConfigAndCheckpointTests(unittest.TestCase):
    def test_config_derives_checkpoint_tag_from_query_variant(self):
        config = importlib.import_module("config")

        self.assertTrue(config.USE_CLIP_IMAGE_QUERY)
        self.assertEqual(config.MODEL_VERSION, "v11")
        self.assertEqual(config.QUERY_VARIANT, "clip_image_query")
        self.assertEqual(config.CHECKPOINT_TAG, "v11_clip_image_query")

    def test_checkpoint_metadata_rejects_query_variant_mismatch(self):
        inference = importlib.import_module("test")
        intent_generator = nn.Identity()
        checkpoint = {
            "model_version": "v11",
            "use_clip_image_query": False,
            "intent_generator_type": "Identity",
        }

        with self.assertRaisesRegex(RuntimeError, "query variant mismatch"):
            inference.validate_checkpoint_metadata(checkpoint, intent_generator)

    def test_network_exports_both_intent_variants(self):
        network = importlib.import_module("net.Network")

        self.assertTrue(hasattr(network, "DualDomainTextIntentGenerator"))
        self.assertTrue(hasattr(network, "DualStreamIntentMLP"))
