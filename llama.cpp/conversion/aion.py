from __future__ import annotations

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("AionInstructForCausalLM")
@ModelBase.example("microsoft/Aion-1.0-Instruct-2.3B")
class AionModel(TextModel):
    model_arch = gguf.MODEL_ARCH.AION

    def set_gguf_parameters(self) -> None:
        super().set_gguf_parameters()
        hparams = self.hparams

        layer_types: list[str] = hparams["layer_types"]
        if len(layer_types) != hparams["num_hidden_layers"]:
            raise ValueError("aion: layer_types must name every layer")
        if set(layer_types) - {"sliding_attention", "full_attention"}:
            raise ValueError(f"aion: unknown layer type in {sorted(set(layer_types))}")

        # the base writes the YaRN keys from rope_parameters.full_attention and
        # rope.freq_base_swa from rope_parameters.sliding_attention; the runtime applies
        # plain RoPE to the sliding layers, so nothing else may be configured for them
        swa_rope = self.rope_parameters.get("sliding_attention", {})
        if swa_rope.get("rope_type", "default") != "default":
            raise ValueError("aion: only plain RoPE is supported on the sliding-window layers")
        full_rope = self.rope_parameters.get("full_attention", {})
        if full_rope.get("rope_type") == "yarn":
            if full_rope.get("truncate", True) is not True or full_rope.get("attention_factor") is not None:
                raise ValueError("aion: YaRN with truncate=false or an explicit attention_factor is not supported")

        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in layer_types])
        logger.info(f"gguf: {layer_types.count('sliding_attention')} of {len(layer_types)} layers use the sliding window")
