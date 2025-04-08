# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature
from sparse_autoencoder import SparseAutoencoder
from sparse_autoencoder.autoencoder.components import EncoderBlock, DecoderBlock
from sparse_autoencoder.loss import EntroppyScheduler, SparsityLoss

import gr00t
from gr00t.model.backbone.eagle2_hg_model.inference_eagle_repo import (
    reshape_model_embeddings,
)

from .eagle2_hg_model.inference_eagle_repo import EagleProcessor, ModelSpecificValues

DEFAULT_EAGLE_MODEL_NAME = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)


def get_embeddings(
    self,
    reproject_vision: bool,
    pixel_values=None,
    input_ids=None,
    attention_mask=None,
    visual_features=None,
    output_hidden_states=None,
    skip_llm=False,
) -> torch.LongTensor:
    assert self.img_context_token_id is not None
    assert pixel_values is not None
    if visual_features is not None:
        vit_embeds = visual_features
    else:
        vit_embeds = self.extract_feature(pixel_values)

    input_embeds = self.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    input_ids = input_ids.reshape(B * N)
    selected = input_ids == self.img_context_token_id
    assert selected.sum() != 0

    embeds_to_scatter = vit_embeds.reshape(-1, C).to(input_embeds.device, input_embeds.dtype)
    input_embeds[selected] = embeds_to_scatter
    input_embeds = input_embeds.reshape(B, N, C)

    # return hidden_states
    embeddings = self.language_model.forward(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    embeddings = embeddings.hidden_states[-1]

    if reproject_vision:
        embeddings = embeddings.reshape(B * N, C)
        embeddings[selected] = embeds_to_scatter
        embeddings = embeddings.reshape(B, N, C)
    return embeddings


class EagleBackbone(nn.Module):
    def __init__(
        self,
        select_layer: int = 12,
        model_name: str = DEFAULT_EAGLE_MODEL_NAME,
        tune_llm: bool = False,
        tune_visual: bool = False,
        reproject_vision: bool = False,
        scale_image_resolution: int = 1,
        processor_cfg: dict = None,
        projector_dim: int = -1,
        allow_reshape_visual: bool = True,
        remove_llm: bool = False,
        load_pretrained_det_eagle_path=None,
        use_local_eagle_hg_model: bool = True,
        use_sae: bool = False,
        sae_cfg: dict = None,
    ):
        super().__init__()
        self.reproject_vision = reproject_vision

        # use local eagle model
        if use_local_eagle_hg_model:
            model_name = DEFAULT_EAGLE_MODEL_NAME

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_config(config, trust_remote_code=True)
        self.model.neftune_alpha = None

        if hasattr(self.model.vision_model, "vision_model") and hasattr(
            self.model.vision_model.vision_model, "head"
        ):
            self.model.vision_model.vision_model.head = torch.nn.Identity()

        # remove parts of the LLM
        self.model.language_model.lm_head = torch.nn.Identity()
        while len(self.model.language_model.model.layers) > select_layer:
            self.model.language_model.model.layers.pop(-1)

        # initialize processor
        processor = EagleProcessor(
            model_path=processor_cfg["model_path"],
            max_input_tiles=processor_cfg["max_input_tiles"],
            model_spec=ModelSpecificValues(**processor_cfg["model_spec"]),
        )
        self.model.img_context_token_id = processor.get_img_context_token()

        assert self.model.template == processor.model_spec.template
        assert self.model.num_image_token == processor.model_spec.num_image_token

        if projector_dim != -1:
            self.linear = torch.nn.Linear(projector_dim, 1536)
        else:
            self.linear = torch.nn.Identity()

        if allow_reshape_visual and scale_image_resolution != 1:
            reshape_model_embeddings(self.model, scale_image_resolution)

        if load_pretrained_det_eagle_path is not None:
            print("loading eagle model weight from: {}".format(load_pretrained_det_eagle_path))
            self.model.load_state_dict(torch.load(load_pretrained_det_eagle_path))

        self.set_trainable_parameters(tune_llm, tune_visual)

        if (
            hasattr(self.model, "vision_model")
            and hasattr(self.model.vision_model, "vision_model")
            and hasattr(self.model.vision_model.vision_model, "vision_towers")
            and len(self.model.vision_model.vision_model.vision_towers) > 1
        ):
            vision_towers = self.model.vision_model.vision_model.vision_towers

            if (
                hasattr(vision_towers[0], "vision_tower")
                and hasattr(vision_towers[0].vision_tower, "vision_model")
                and hasattr(vision_towers[0].vision_tower.vision_model, "encoder")
            ):
                vision_towers[0].vision_tower.vision_model.encoder.gradient_checkpointing = False
                vision_towers[0].vision_tower.vision_model.head = torch.nn.Identity()

            if hasattr(vision_towers[1], "vision_tower"):
                vision_towers[1].vision_tower.head = torch.nn.Identity()
             
                
        # Add Sparse Autoencoder
        self.use_sae = use_sae
        if use_sae:
            self.sae = SparseAutoencoder(
                n_input_features=1536,  # Match your embedding dimension
                n_learned_features=sae_cfg.get("n_learned_features", 3072),  # 2x expansion
                encoder=EncoderBlock(
                    n_input_features=1536,
                    n_learned_features=sae_cfg.get("n_learned_features", 3072),
                    bias=True,
                ),
                decoder=DecoderBlock(
                    n_learned_features=sae_cfg.get("n_learned_features", 3072),
                    n_input_features=1536,
                    bias=True,
                ),
            )
            
            self.entropy_scheduler = EntropyScheduler(
                initial_entropy=sae_cfg.get("initial_entropy", 3.0),
                final_entropy=sae_cfg.get("final_entropy", 2.0),
                n_steps=sae_cfg.get("n_entropy_steps", 1000)
            )
            
            self.sparsity_loss = SparsityLoss(
                l1_coefficient=sae_cfg.get("l1_coefficient", 1e-3)
            )
    
    # Add to EagleBackbone class

    def get_sae_features(self, embeddings):
        """Extract and analyze SAE features"""
        if not self.use_sae:
            raise ValueError("SAE not enabled")
            
        with torch.no_grad():
            encoded = self.sae.encode(embeddings)
            
        # Get feature statistics
        feature_means = encoded.mean(dim=0)
        feature_stds = encoded.std(dim=0)
        sparsity = (encoded == 0).float().mean(dim=0)
        
        return {
            "activations": encoded,
            "means": feature_means,
            "stds": feature_stds,
            "sparsity": sparsity
        }

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        if not tune_llm:
            self.model.language_model.requires_grad_(False)
        if not tune_visual:
            self.model.vision_model.requires_grad_(False)
            self.model.mlp1.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if self.model.vision_model and not self.tune_visual:
                self.model.vision_model.eval()
                self.model.mlp1.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        # 0. Set frozen module to eval
        self.set_frozen_modules_to_eval_mode()

        embeddings = get_embeddings(
            self.model,
            self.reproject_vision,
            pixel_values=vl_input["pixel_values"],
            input_ids=vl_input["input_ids"],
            attention_mask=vl_input["attention_mask"],
        )

       # Apply SAE if enabled
        if self.use_sae and self.training:
            # Get activations and reconstruction
            encoded_activations = self.sae.encode(embeddings)
            reconstructed = self.sae.decode(encoded_activations)
            
            # Calculate losses
            recon_loss = torch.nn.functional.mse_loss(reconstructed, embeddings)
            l1_loss = self.sparsity_loss(encoded_activations)
            
            # Store losses for logging
            self.sae_losses = {
                "reconstruction_loss": recon_loss.item(),
                "l1_loss": l1_loss.item(),
            }
            
            # Use reconstructed embeddings
            embeddings = reconstructed
        elif self.use_sae:
            # Inference mode - just encode and decode
            embeddings = self.sae.decode(self.sae.encode(embeddings))

        embeddings = self.linear(embeddings)

        attention_mask = vl_input["attention_mask"]
        return BatchFeature(
            data={
                "backbone_features": embeddings,
                "backbone_attention_mask": attention_mask,
            }
        )  # [B, T2, hidden_size]
