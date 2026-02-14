from __future__ import annotations

from typing import Optional, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==============================================================================
# LLM WRAPPER
# ==============================================================================

class LLMWrapper:
    """Wrapper for a HuggingFace causal LM with chat template, optional LoRA, and device selection (cuda > mps > cpu)."""

    def __init__(
        self,
        model_name: str,
        lora_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_name = model_name
        self.lora_path = lora_path

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        if dtype is None:
            if self.device in ("cuda", "mps"):
                dtype = torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                **({k: v for k, v in model_kwargs.items() if k != "torch_dtype"}),
            )
        except TypeError:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                **model_kwargs,
            )
        if self.device != "cuda":
            base_model.to(self.device)

        if self.lora_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base_model, self.lora_path)
        else:
            self.model = base_model

        self.model.eval()

    def _build_inputs(self, prompt: str) -> tuple[torch.Tensor, int]:
        """Build input_ids and return input length for trimming generated output. Uses chat template when available."""
        has_chat_template = (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
            and str(self.tokenizer.chat_template).strip() != ""
        )

        if has_chat_template:
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
        else:
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"]

        input_ids = input_ids.to(self.device)
        input_len = input_ids.shape[-1]
        return input_ids, input_len

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: Optional[bool] = None,
    ) -> str:
        """Generate text from prompt. Returns only the continuation, not the prompt."""
        if do_sample is None:
            do_sample = temperature > 0.0

        input_ids, input_len = self._build_inputs(prompt)

        generated_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen_only = generated_ids[0, input_len:]
        text = self.tokenizer.decode(gen_only, skip_special_tokens=True)

        return text.strip()

    @torch.inference_mode()
    def generate_answer_only(
        self,
        prompt: str,
        extractor_fn: Callable[[str], str],
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        """Generate output and apply extractor to get the desired part (e.g. content of <answer>)."""
        text = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return extractor_fn(text)