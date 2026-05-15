import torch
from models.llm_llama2 import LLaMa2LLMBackbone
from config.config_vlm import Prism_7B_DINOSigLIP_224px
import time


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = Prism_7B_DINOSigLIP_224px()
    hf_token = "your hf token here"
    start_time = time.time()
    llm_backbone_pretrained = LLaMa2LLMBackbone(config=config, hf_token=hf_token, use_flash_attention_2=False, load_pretrained=True).to(device)
    pretrained_time = time.time() - start_time
    print(f"Time taken to load pretrained model: {pretrained_time:.2f} seconds")

    # start_time = time.time()
    # llm_backbone_non_pretrained = LLaMa2LLMBackbone(config=config, hf_token=hf_token, use_flash_attention_2=False, load_pretrained=False).to(device)
    # non_pretrained_time = time.time() - start_time
    # print(f"Time taken to load non-pretrained model: {non_pretrained_time:.2f} seconds")

    llm_backbone = llm_backbone_pretrained
    tokenizer = llm_backbone.get_tokenizer()
    prompt_builder = llm_backbone.prompt_builder

    prompt_builder.add_turn("human", "Hello, how are you?")
    prompt = prompt_builder.get_prompt()

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(inputs["input_ids"].shape)

    with torch.no_grad():
        outputs = llm_backbone(**inputs)

    print("Prompt:", prompt)
    print("Logits shape:", outputs.logits.shape)
    print(
        "pad/bos/eos token ids:",
        llm_backbone.pad_token_id,
        llm_backbone.bos_token_id,
        llm_backbone.eos_token_id,
    )

