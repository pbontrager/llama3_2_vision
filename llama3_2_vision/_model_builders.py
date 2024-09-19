# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torchtune.models.flamingo import flamingo_decoder, flamingo_vision_encoder, FlamingoTransform
from torchtune.modules.model_fusion import DeepFusionModel
from torchtune.modules.tokenizers import parse_hf_tokenizer_json
from torchtune.data._prompt_templates import _TemplateType
from torchtune.data._utils import _get_prompt_template


def llama3_2_vision_11b(decoder_trainable=False, encoder_trainable=True, fusion_trainable=True) -> DeepFusionModel:
    """ Llama 3.2 Vision 11B model

    Args:
        decoder_trainable (bool): Whether to make decoder params trainable. Default is False.
        encoder_trainable (bool): Whether to make encoder params trainable. Default is True.
        fusion_trainable (bool): Whether to make fusion params trainable. Default is True.

    Returns:
        DeepFusionModel: Instantiation of the Llama 3.2 Vision 11B model
    """
    encoder = flamingo_vision_encoder(
        patch_size=14,
        num_heads=16,
        clip_embed_dim=1280,
        clip_num_layers=32,
        clip_hidden_states=[3, 7, 15, 23, 30],
        decoder_embed_dim=4096,
        num_layers_projection=8,
        tile_size=560,
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
        encoder_max_seq_len=64040,
        rope_base=500000.0,
        intermediate_dim=14336,
    )
    return DeepFusionModel(
        encoder=encoder,
        decoder=decoder,
        encoder_trainable=encoder_trainable,
        decoder_trainable=decoder_trainable,
        fusion_trainable=fusion_trainable,
    )


def llama3_2_vision_transform(path: str, max_seq_len: int = 8192, special_tokens_path: Optional[str] = None, prompt_template: Optional[_TemplateType] = None) -> FlamingoTransform:
    """
    Data Transforms (including Tokenizer) for Llama3 Vision.

    Args:
        path (str): path to the tokenizer
        max_seq_len (int): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file 
            structured similarly. Default is None to use the canonical Llama3 special tokens.
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags.
    
    Returns:
        FlamingoTransform: Instantiation of the Llama3 tokenizer
    """
    special_tokens = parse_hf_tokenizer_json(special_tokens_path) if special_tokens_path is not None else None
    template = _get_prompt_template(prompt_template) if prompt_template is not None else None
    return FlamingoTransform(
        path=path,
        special_tokens=special_tokens,
        tile_size=560,
        patch_size=14,
        max_num_tiles=4,
        max_seq_len=max_seq_len,
        image_mean=(0.48145466, 0.4578275, 0.40821073),
        image_std=(0.26862954, 0.26130258, 0.27577711),
        prompt_template=template,
    )
