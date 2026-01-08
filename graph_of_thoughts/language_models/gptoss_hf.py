import os
import torch
from typing import List, Dict
from .abstract_language_model import AbstractLanguageModel


class GPTOssHF(AbstractLanguageModel):
    """
    An interface to use OpenAI GPT-OSS models through the HuggingFace Transformers library.

    Expected HF model id example: "openai/gpt-oss-20b"
    """

    def __init__(
        self, config_path: str = "", model_name: str = "gpt-oss-20b", cache: bool = False
    ) -> None:
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]

        # Hugging Face repo id, e.g., "openai/gpt-oss-20b"
        self.model_id: str = self.config["model_id"]

        # Costs for 1000 tokens (kept for compatibility; you may keep them 0)
        self.prompt_token_cost: float = self.config.get("prompt_token_cost", 0.0)
        self.response_token_cost: float = self.config.get("response_token_cost", 0.0)

        # Generation params
        self.temperature: float = float(self.config.get("temperature", 0.0))
        self.top_k: int = int(self.config.get("top_k", 0))  # 0 means "disable top-k"
        # In your original class, max_tokens was passed as max_length.
        # For chat-template generation, it is more reliable to use max_new_tokens.
        self.max_new_tokens: int = int(self.config.get("max_tokens", 256))

        # Cache dir (HF_HOME is the recommended env var)
        cache_dir = self.config.get("cache_dir", "")
        if cache_dir:
            os.environ["HF_HOME"] = cache_dir
            # If you previously depended on TRANSFORMERS_CACHE, you can optionally keep it:
            # os.environ["TRANSFORMERS_CACHE"] = cache_dir

        # Optional: if your server had issues with Xet, allow disabling via config
        # Example config key: "disable_xet": true
        if bool(self.config.get("disable_xet", False)):
            os.environ["HF_HUB_DISABLE_XET"] = "1"

        import transformers

        # Load tokenizer/model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        # Ensure pad token exists (some chat models rely on eos for padding)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def query(self, query: str, num_responses: int = 1) -> List[Dict]:
        """
        Query GPT-OSS via HF Transformers.

        Returns: List[Dict] with {"generated_text": "..."} for compatibility with existing parsers.
        """
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        # Build chat messages. Keep system prompt short and format-enforcing.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Follow instructions precisely. "
                    "Output only what is requested, in the exact required format."
                ),
            },
            {"role": "user", "content": query},
        ]

        # Convert messages -> input_ids using the model's chat template
        # add_generation_prompt=True adds the assistant prefix if the template requires it.
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generation config
        do_sample = self.temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "temperature": self.temperature if do_sample else None,
            "top_k": self.top_k if (do_sample and self.top_k and self.top_k > 0) else None,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        # Remove None values to avoid transformers warnings
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        outputs = []
        for _ in range(num_responses):
            out = self.model.generate(inputs, **gen_kwargs)
            # Slice off the prompt tokens
            gen_ids = out[0][inputs.shape[-1] :]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            outputs.append({"generated_text": text})

        if self.cache:
            self.response_cache[query] = outputs
        return outputs

    def get_response_texts(self, query_responses: List[Dict]) -> List[str]:
        return [qr["generated_text"] for qr in query_responses]