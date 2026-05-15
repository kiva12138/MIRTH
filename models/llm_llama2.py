from typing import List, Optional, Sequence, Tuple, Type
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizerBase,
    AutoConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from config.config_vlm import Prism_7B_DINOSigLIP_224px


class LLaMA2PurePromptBuilder():
    def __init__(self) -> None:
        self.bos, self.eos = "<s>", "</s>"

        # Get role-specific "wrap" functions
        self.wrap_human = lambda msg: f"In: {msg}\nOut: "
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        if (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        human_message = self.wrap_human(message)
        prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        # Remove prefix <bos> (if exists) because it gets auto-inserted by tokenizer!
        return self.prompt.removeprefix(self.bos).rstrip()


class LLaMa2LLMBackbone(nn.Module):
    def __init__(self, config, hf_token, use_flash_attention_2, load_pretrained=True) -> None:
        super().__init__()
        self.config = config
        self.llm_max_length = config.llm_max_length

        if load_pretrained:
            # This will load the LLaMA-2 model weights twice
            # But I don't know where this is faster than loading the model without weights and then loading the weights separately, so... ¯\_(ツ)_/¯
            self.llm = LlamaForCausalLM.from_pretrained(
                config.llm_backbone_hf_id,
                token=hf_token,
                attn_implementation="flash_attention_2" if use_flash_attention_2 else "sdpa",
                dtype=self.half_precision_dtype,
            )
        else:
            llm_config = AutoConfig.from_pretrained(config.llm_backbone_hf_id, token=hf_token)
            self.llm = LlamaForCausalLM._from_config(llm_config, dtype=self.half_precision_dtype, attn_implementation="flash_attention_2" if use_flash_attention_2 else "sdpa",)
        
        self.llm.config.use_cache = False
        self.llm.enable_input_require_grads()

        self.tokenizer:PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            config.llm_backbone_hf_id,
            model_max_length=self.llm_max_length,
            token=hf_token,
            padding_side="right",
        )
        
        self.llama_prompt_builder = LLaMA2PurePromptBuilder()
        self.check_padding_side()

        # [Special Case] LLaMa-2 PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    def check_padding_side(self):
        assert self.tokenizer.padding_side == "right", "Tokenizer `padding_side` is not set to `right`!"
        assert (self.tokenizer("Test 123", add_special_tokens=True).input_ids[0] == self.tokenizer.bos_token_id) and (
            self.tokenizer("Test 123", add_special_tokens=False).input_ids[0] != self.tokenizer.bos_token_id), (
            f"Default Tokenizer of type `{type(self.tokenizer)}` does not automatically prefix inputs with BOS token!\n"
            "Please read the comment in `base_llm.py` for more information!")

    def enable_gradient_checkpointing(self) -> None:
        self.llm.gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self) -> None:
        self.llm.gradient_checkpointing_disable()

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        output: CausalLMOutputWithPast = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return output

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer
    
    @property
    def prompt_builder(self):
        return self.llama_prompt_builder

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return LlamaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def embed_dim(self) -> int:
        return self.llm.config.hidden_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id
    
    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_token(self) -> int:
        return self.tokenizer.pad_token

    @property
    def bos_token(self) -> int:
        return self.tokenizer.bos_token

    @property
    def eos_token(self) -> int:
        return self.tokenizer.eos_token
    
    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        return (self.llm.model.embed_tokens, self.llm.model.layers[-1], self.llm.lm_head)


def get_Prism_7B_DINOSigLIP_224px_backbone_llama2_and_tokenizer(
    hf_token: Optional[str],
    use_flash_attention_2: bool,
    load_pretrained: bool = True,
) -> Tuple[LLaMa2LLMBackbone, PreTrainedTokenizerBase]:
    config = Prism_7B_DINOSigLIP_224px()
    llm_backbone: LLaMa2LLMBackbone = LLaMa2LLMBackbone(
        config,
        hf_token,
        use_flash_attention_2,
        load_pretrained=load_pretrained,
    )
    tokenizer: PreTrainedTokenizerBase = llm_backbone.get_tokenizer()

    return llm_backbone, tokenizer
