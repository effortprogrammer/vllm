# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, cast

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BatchFeature, Gemma4Config, Gemma4Processor

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageProcessorItems, ImageSize, MultiModalDataItems
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)


class Gemma4ImagePatchInputs(TensorSchema):
    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[torch.Tensor, TensorShape("bn", "p", "d")]
    image_position_ids: Annotated[torch.Tensor, TensorShape("bn", "p", 2)]
    num_soft_tokens_per_image: Annotated[torch.Tensor, TensorShape("bn")]


class Gemma4ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Gemma4Config)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(Gemma4Processor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_hf_processor()
        image_processor = processor.image_processor
        max_patches = (
            image_processor.max_soft_tokens * image_processor.pooling_kernel_size**2
        )
        width_patches = int(math.sqrt(max_patches))
        while max_patches % width_patches != 0:
            width_patches -= 1
        height_patches = max_patches // width_patches
        patch_size = image_processor.patch_size
        return ImageSize(
            height=height_patches * patch_size, width=width_patches * patch_size
        )


class Gemma4DummyInputsBuilder(BaseDummyInputsBuilder[Gemma4ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        image_token = self.info.get_hf_processor().image_token
        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        target_width, target_height = self.info.get_image_size_with_most_features()
        image_overrides = cast(Any, mm_options.get("image"))
        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class Gemma4MultiModalProcessor(BaseMultiModalProcessor[Gemma4ProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "image_position_ids": MultiModalFieldConfig.batched("image"),
            "num_soft_tokens_per_image": MultiModalFieldConfig.batched("image"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_token = hf_processor.image_token
        images = mm_items.get_items("image", ImageProcessorItems)

        def get_replacement(item_idx: int):
            image_size = images.get_image_size(item_idx)
            mm_token_info = hf_processor._get_num_multimodal_tokens(
                image_sizes=[(image_size.height, image_size.width)]
            )
            num_soft_tokens = int(mm_token_info.num_image_tokens[0])
            repl = (
                hf_processor.boi_token
                + hf_processor.image_token * num_soft_tokens
                + hf_processor.eoi_token
            )
            return PromptUpdateDetails.select_text(repl, hf_processor.image_token)

        return [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_replacement,
            )
        ]


class Gemma4MultimodalEmbedder(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.embedding_projection = nn.Linear(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            bias=False,
        )
        self.embedding_pre_projection_norm = RMSNorm(
            config.vision_config.hidden_size,
            eps=config.vision_config.rms_norm_eps,
            has_weight=False,
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(embs_normed)


class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        vision_config = config.vision_config
        assert vision_config is not None
        self.hidden_size = vision_config.hidden_size
        self.patch_size = vision_config.patch_size
        self.position_embedding_size = vision_config.position_embedding_size
        self.input_proj = nn.Linear(
            3 * self.patch_size**2, self.hidden_size, bias=False
        )
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, self.position_embedding_size, self.hidden_size)
        )

    def _position_embeddings(
        self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor
    ) -> torch.Tensor:
        clamped_positions = pixel_position_ids.clamp(min=0)
        one_hot = F.one_hot(clamped_positions, num_classes=self.position_embedding_size)
        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)
        position_embeddings = (one_hot @ self.position_embedding_table).sum(dim=1)
        return torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        pixel_values = 2 * (pixel_values - 0.5)
        hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        return hidden_states + self._position_embeddings(
            image_position_ids, padding_positions
        )


class Gemma4VisionRotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        vision_config = config.vision_config
        assert vision_config is not None
        self.head_dim = vision_config.head_dim
        spatial_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            vision_config.rope_parameters["rope_theta"]
            ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32) / spatial_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        all_cos = []
        all_sin = []
        for i in range(2):
            dim_position_ids = position_ids[:, :, i][:, None, :].float()
            freqs = (inv_freq.to(x.device) @ dim_position_ids.to(x.device)).transpose(
                1, 2
            )
            emb = torch.cat((freqs, freqs), dim=-1)
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())
        cos = torch.cat(all_cos, dim=-1).to(dtype=x.dtype)
        sin = torch.cat(all_sin, dim=-1).to(dtype=x.dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (_rotate_half(x) * sin)


def _apply_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    ndim = position_ids.shape[-1]
    num_input_channels = x.shape[-1]
    num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))
    split_sizes = [num_rotated_channels_per_dim] * ndim
    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    y_parts = [
        _apply_rotary(
            x_parts[k], cos_parts[k], sin_parts[k], unsqueeze_dim=unsqueeze_dim
        )
        for k in range(ndim)
    ]
    return torch.cat(y_parts, dim=-1)


class Gemma4VisionAttention(nn.Module):
    def __init__(self, config: Gemma4Config, prefix: str = ""):
        super().__init__()
        vision_config = config.vision_config
        assert vision_config is not None
        self.num_heads = vision_config.num_attention_heads
        self.num_kv_heads = vision_config.num_key_value_heads
        self.head_dim = vision_config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = 1.0
        self.attention_dropout = float(vision_config.attention_dropout or 0.0)
        self.q_proj = nn.Linear(
            vision_config.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            vision_config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            vision_config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            vision_config.hidden_size,
            bias=False,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=vision_config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=vision_config.rms_norm_eps)
        self.v_norm = RMSNorm(
            self.head_dim, eps=vision_config.rms_norm_eps, has_weight=False
        )

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
        if self.num_key_value_groups == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, self.num_key_value_groups, seq_len, head_dim
        )
        return hidden_states.reshape(
            batch, num_key_value_heads * self.num_key_value_groups, seq_len, head_dim
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)
        cos, sin = position_embeddings

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        query_states = _apply_multidimensional_rope(
            query_states, cos, sin, position_ids
        ).transpose(1, 2)

        key_states = self.k_norm(
            self.k_proj(hidden_states).view(
                batch_size, seq_len, self.num_kv_heads, self.head_dim
            )
        )
        key_states = _apply_multidimensional_rope(
            key_states, cos, sin, position_ids
        ).transpose(1, 2)
        value_states = self.v_norm(
            self.v_proj(hidden_states).view(
                batch_size, seq_len, self.num_kv_heads, self.head_dim
            )
        ).transpose(1, 2)

        key_states = self._repeat_kv(key_states)
        value_states = self._repeat_kv(value_states)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = (
            attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1).contiguous()
        )
        return self.o_proj(attn_output)


class Gemma4VisionMLP(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        vision_config = config.vision_config
        assert vision_config is not None
        self.gate_proj = nn.Linear(
            vision_config.hidden_size,
            vision_config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            vision_config.hidden_size,
            vision_config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            vision_config.intermediate_size,
            vision_config.hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


class Gemma4VisionEncoderLayer(nn.Module):
    def __init__(self, config: Gemma4Config, prefix: str = ""):
        super().__init__()
        vision_config = config.vision_config
        assert vision_config is not None
        self.self_attn = Gemma4VisionAttention(
            config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Gemma4VisionMLP(config)
        self.input_layernorm = RMSNorm(
            vision_config.hidden_size, eps=vision_config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            vision_config.hidden_size, eps=vision_config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            vision_config.hidden_size, eps=vision_config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            vision_config.hidden_size, eps=vision_config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask,
            position_ids,
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.post_feedforward_layernorm(hidden_states)
        return hidden_states


class Gemma4VisionPooler(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        vision_config = config.vision_config
        assert vision_config is not None
        self.root_hidden_size = vision_config.hidden_size**0.5

    def _avg_pool_by_positions(
        self,
        hidden_states: torch.Tensor,
        image_position_ids: torch.Tensor,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k**2
        clamped_positions = image_position_ids.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        if hidden_states.shape[1] != output_length:
            hidden_states, padding_positions = self._avg_pool_by_positions(
                hidden_states, image_position_ids, output_length
            )
        hidden_states *= self.root_hidden_size
        return hidden_states, padding_positions


class Gemma4VisionModel(nn.Module):
    def __init__(self, config: Gemma4Config, prefix: str = ""):
        super().__init__()
        vision_config = config.vision_config
        assert vision_config is not None
        self.config = config
        self.patch_embedder = Gemma4VisionPatchEmbedder(config)
        self.rotary_emb = Gemma4VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [
                Gemma4VisionEncoderLayer(
                    config,
                    prefix=f"{prefix}.encoder.layers.{i}",
                )
                for i in range(vision_config.num_hidden_layers)
            ]
        )
        self.pooler = Gemma4VisionPooler(config)
        self.std_bias = nn.Parameter(
            torch.zeros(vision_config.hidden_size), requires_grad=False
        )
        self.std_scale = nn.Parameter(
            torch.ones(vision_config.hidden_size), requires_grad=False
        )

    def _build_attention_mask(self, valid_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = valid_mask[:, None, :, None] & valid_mask[:, None, None, :]
        attn_mask = torch.where(
            attn_mask,
            torch.zeros((), device=valid_mask.device),
            torch.full(
                (),
                torch.finfo(torch.float32).min,
                device=valid_mask.device,
            ),
        )
        return attn_mask

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        pooling_kernel_size = self.config.vision_config.pooling_kernel_size
        output_length = pixel_values.shape[1] // (
            pooling_kernel_size * pooling_kernel_size
        )
        padding_positions = (image_position_ids == -1).all(dim=-1)
        hidden_states = self.patch_embedder(
            pixel_values, image_position_ids, padding_positions
        )
        position_embeddings = self.rotary_emb(hidden_states, image_position_ids)
        attention_mask = self._build_attention_mask(~padding_positions)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings,
                attention_mask,
                image_position_ids,
            )
        hidden_states, pooler_mask = self.pooler(
            hidden_states,
            image_position_ids,
            padding_positions,
            output_length,
        )
        hidden_states = hidden_states[pooler_mask]
        if self.config.vision_config.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    Gemma4MultiModalProcessor,
    info=Gemma4ProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class Gemma4ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA
):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.embed_vision.": "embed_vision.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|image|>"
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config

        self.configure_mm_token_handling(
            vocab_size=config.text_config.vocab_size,
            mm_token_ids=[config.image_token_id],
        )

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = Gemma4VisionModel(
                config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.embed_vision = Gemma4MultimodalEmbedder(config)

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Gemma4ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Gemma4ImagePatchInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_position_ids = kwargs.pop("image_position_ids", None)
        num_soft_tokens_per_image = kwargs.pop("num_soft_tokens_per_image", None)
        if (
            pixel_values is None
            or image_position_ids is None
            or num_soft_tokens_per_image is None
        ):
            return None
        return Gemma4ImagePatchInputs(
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            num_soft_tokens_per_image=num_soft_tokens_per_image,
        )

    def _process_image_input(
        self, image_input: Gemma4ImagePatchInputs
    ) -> list[torch.Tensor]:
        image_features = self.vision_tower(
            image_input["pixel_values"],
            image_input["image_position_ids"],
        )
        image_embeds = self.embed_vision(image_features)
        return list(
            image_embeds.split(image_input["num_soft_tokens_per_image"].tolist())
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        return self._process_image_input(image_input)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)
        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(
            weights=weights,
            mapper=self.hf_to_vllm_mapper,
        )

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="embed_vision",
            tower_model="vision_tower",
        )

    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int:
        return num_image_tokens

    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int:
        return num_vision_tokens
