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

from typing import List, Optional, Tuple

import torch

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import MiniTrainDIT


class MinimalV1LVGDiT(MiniTrainDIT):
    def __init__(self, *args, timestep_scale: float = 1.0, reference_channels: int = 0, **kwargs):
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += 1 + reference_channels
        self.timestep_scale = timestep_scale
        self.reference_channels = reference_channels
        super().__init__(*args, **kwargs)
        if reference_channels > 0:
            self._zero_init_reference_channels()

    def _zero_init_reference_channels(self):
        """Zero-initialize the x_embedder weights for reference_latent channels.

        After construction the PatchEmbed linear has shape (D, total_ch * patch_vol).
        We zero out columns corresponding to reference channels so the model
        starts training as if no reference input exists (preserving pretrained behavior).

        Channel layout in x_embedder input:
          [noisy_latent(16), reference_latent(ref_ch), condition_mask(1), padding_mask(1)]
        self.in_channels = 16 + ref_ch + 1 (padding_mask is added by build_patch_embed)
        """
        linear = self.x_embedder.proj[1]  # nn.Linear inside PatchEmbed.proj Sequential
        patch_vol = self.patch_temporal * self.patch_spatial ** 2
        base_noisy_ch = self.in_channels - self.reference_channels - 1  # = original in_channels (16)
        ref_start = base_noisy_ch * patch_vol
        ref_end = (base_noisy_ch + self.reference_channels) * patch_vol
        with torch.no_grad():
            linear.weight[:, ref_start:ref_end].zero_()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Handle x_embedder weight remapping when loading a pretrained checkpoint
        that was saved without reference channels."""
        if self.reference_channels > 0:
            embedder_key = "x_embedder.proj.1.weight"
            if embedder_key in state_dict:
                old_weight = state_dict[embedder_key]
                new_weight = self.x_embedder.proj[1].weight
                if old_weight.shape != new_weight.shape:
                    patch_vol = self.patch_temporal * self.patch_spatial ** 2
                    n_old_in = old_weight.shape[1] // patch_vol
                    noisy_end = (n_old_in - 2) * patch_vol  # noisy latent channels (excl. mask + padding)
                    ref_size = self.reference_channels * patch_vol

                    remapped = torch.zeros_like(new_weight)
                    remapped[:, :noisy_end] = old_weight[:, :noisy_end]
                    remapped[:, noisy_end + ref_size:] = old_weight[:, noisy_end:]
                    state_dict[embedder_key] = remapped
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        intermediate_feature_ids: Optional[List[int]] = None,
        img_context_emb: Optional[torch.Tensor] = None,
        reference_latent_B_C_T_H_W: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs

        if data_type == DataType.VIDEO:
            parts = [x_B_C_T_H_W]
            if self.reference_channels > 0:
                if reference_latent_B_C_T_H_W is not None:
                    parts.append(reference_latent_B_C_T_H_W.type_as(x_B_C_T_H_W))
                else:
                    B, _, T, H, W = x_B_C_T_H_W.shape
                    parts.append(torch.zeros(B, self.reference_channels, T, H, W,
                                             dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device))
            parts.append(condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W))
            x_B_C_T_H_W = torch.cat(parts, dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            zeros_ref = self.reference_channels
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W,
                 torch.zeros((B, zeros_ref, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device) if zeros_ref > 0 else torch.empty(0),
                 torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )
        return super().forward(
            x_B_C_T_H_W=x_B_C_T_H_W,
            timesteps_B_T=timesteps_B_T * self.timestep_scale,
            crossattn_emb=crossattn_emb,
            fps=fps,
            padding_mask=padding_mask,
            data_type=data_type,
            intermediate_feature_ids=intermediate_feature_ids,
            img_context_emb=img_context_emb,
        )
