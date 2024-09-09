# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtune.models.flamingo import flamingo_decoder, flamingo_vision_encoder, FlamingoTransform
from torchtune.modules.model_fusion import DeepFusionModel


def llama_3_2_vision_11b() -> DeepFusionModel:
    encoder = flamingo_vision_encoder(
        patch_size=14,
        num_heads=16,
        clip_embed_dim=1280,
        clip_num_layers=32,
        clip_hidden_states=[3, 7, 15, 23, 30],
        decoder_embed_dim=4096,
        num_layers_projection=8,
        tile_size=448,
        max_num_tiles=4,
        in_channels=3,
    )
    decoder = flamingo_decoder(
        vocab_size=128_256,
        num_layers=32,
        fusion_interval=4,
        num_special_tokens=8,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=8192,
        encoder_max_seq_len=4100,
        rope_base=500000.0,
        intermediate_dim=14336,
    )
    return DeepFusionModel(
        encoder=encoder,
        decoder=decoder,
    )

def llama3_2_vision_transform(tokenizer_path):
    return FlamingoTransform(
        tokenizer_path,
        tile_size=448,
        patch_size=14,
        max_num_tiles=4,
        max_seq_len=8192,
        encoder_max_seq_len=4100,
        image_mean=(0.48145466, 0.4578275, 0.40821073),
        image_std=(0.26862954, 0.26130258, 0.27577711),
        prompt_template=None,
    )
