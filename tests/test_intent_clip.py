import unittest
from unittest import mock

import torch
import torch.nn as nn

from net.frequency_fusion import prompt as prompt_module
from net.intent import (
    DEGRADATION_PROMPT_GROUPS,
    FUSION_PROMPT_GROUPS,
    CLIPImageQuery,
    DualDomainTextIntentGenerator,
    DualStreamIntentMLP,
)


class FakeClipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))

    def encode_text(self, tokens):
        features = torch.zeros(tokens.shape[0], 512, device=tokens.device)
        features.scatter_(1, (tokens[:, :1] % 512).long(), 1.0)
        return features

    def encode_image(self, images):
        features = torch.zeros(images.shape[0], 512, device=images.device)
        features[:, 0] = 1.0
        return features


def fake_tokenize(texts):
    return torch.arange(len(texts), dtype=torch.long).unsqueeze(1)


class ClipIntentTests(unittest.TestCase):
    def test_prompt_bank_aggregates_groups_into_unit_clip_vectors(self):
        bank = prompt_module.CLIPTextPromptBank(FakeClipModel(), DEGRADATION_PROMPT_GROUPS)

        output = bank()

        self.assertEqual(tuple(output.shape), (4, 512))
        self.assertTrue(torch.allclose(output.norm(dim=-1), torch.ones(4), atol=1e-6))
        self.assertFalse(hasattr(bank, "clip_model"))

    def test_clip_image_query_is_frozen_and_stays_in_eval_mode(self):
        query = CLIPImageQuery(FakeClipModel())
        query.train()

        output = query(torch.rand(2, 3, 224, 224))

        self.assertFalse(query.clip_model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in query.clip_model.parameters()))
        self.assertEqual(tuple(output.shape), (2, 512))
        self.assertTrue(torch.allclose(output.norm(dim=-1), torch.ones(2), atol=1e-6))

    def test_generator_rejects_learnable_prompt_mode_before_loading_clip(self):
        with mock.patch("net.intent.clip.load") as load:
            with self.assertRaises(ValueError):
                DualDomainTextIntentGenerator(use_learnable_prompt_embedding=True)
        load.assert_not_called()

    def test_clip_generator_produces_four_way_weights_and_trainable_intents(self):
        fake_clip = FakeClipModel()
        with mock.patch("net.intent.clip.load", return_value=(fake_clip, None)), mock.patch.object(
            prompt_module.clip, "tokenize", side_effect=fake_tokenize
        ):
            generator = DualDomainTextIntentGenerator(intent_dim=7, clip_device="cpu")
            generator.train()
            i_deg, i_fus, aux = generator(torch.rand(2, 3, 224, 224))
            (i_deg.sum() + i_fus.sum()).backward()

        self.assertEqual(tuple(i_deg.shape), (2, 7))
        self.assertEqual(tuple(i_fus.shape), (2, 7))
        self.assertEqual(tuple(aux["deg_prompt_weight"].shape), (2, 4))
        self.assertEqual(tuple(aux["fus_prompt_weight"].shape), (2, 4))
        self.assertTrue(torch.allclose(aux["deg_prompt_weight"].sum(dim=-1), torch.ones(2)))
        self.assertEqual(tuple(aux["logit_scale_deg"].shape), (1,))
        self.assertIsNotNone(generator.deg_proj.weight.grad)
        self.assertIsNotNone(generator.fus_proj.weight.grad)
        self.assertIsNotNone(generator.logit_scale_deg.grad)
        self.assertIsNotNone(generator.logit_scale_fus.grad)
        self.assertFalse(generator.clip_image_query.clip_model.training)

    def test_shared_encoder_mlp_ablation_uses_same_call_signature_without_clip_weights(self):
        fake_clip = FakeClipModel()
        with mock.patch("net.intent.clip.load", return_value=(fake_clip, None)), mock.patch.object(
            prompt_module.clip, "tokenize", side_effect=fake_tokenize
        ):
            generator = DualStreamIntentMLP(channels=1, intent_dim=5, hidden_dim=8, clip_device="cpu")
            spatial = [torch.rand(2, 1, 16, 16), torch.rand(2, 1, 8, 8), torch.rand(2, 1, 4, 4)]
            i_deg, i_fus, aux = generator(torch.rand(2, 3, 224, 224), spatial, spatial, spatial[0], spatial[0])

        self.assertEqual(tuple(i_deg.shape), (2, 5))
        self.assertEqual(tuple(i_fus.shape), (2, 5))
        self.assertEqual(tuple(aux["deg_prompt_weight"].shape), (2, 4))
        self.assertFalse(hasattr(generator, "clip_model"))
        self.assertFalse(any("clip_model" in key for key in generator.state_dict()))
        self.assertEqual(len(FUSION_PROMPT_GROUPS), 4)
