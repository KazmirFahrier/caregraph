"""Production LoRA / QLoRA fine-tuning (GPU path).

This is the real implementation the offline `LogisticExtractor` stands in for: it
fine-tunes a small open LLM (e.g. Llama-3.x-8B-Instruct or Phi-3-mini) on the
problem-list extraction dataset using 4-bit QLoRA via `bitsandbytes` + `peft` +
`trl`'s `SFTTrainer`.

It is import-guarded so the rest of CareGraph installs and tests with no ML stack.
Install the extra and run on a CUDA box:

    pip install "caregraph[finetune]"
    python -m caregraph.finetune.lora --data data/problems.jsonl --base microsoft/Phi-3-mini-4k-instruct

Design choices worth noting in an interview:
- QLoRA (4-bit NF4) so an 8B model fine-tunes on a single 24GB GPU.
- LoRA on attention + MLP projections; rank 16, alpha 32 — a standard, well-behaved
  starting point.
- The adapter is tiny (~tens of MB) and versioned separately from the base weights,
  so you ship a diff, not a 16GB checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QLoRAConfig:
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    output_dir: str = "artifacts/adapter"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    learning_rate: float = 2e-4
    num_epochs: int = 3
    per_device_batch_size: int = 4
    max_seq_len: int = 1024


def train_lora(data_jsonl: str, config: QLoRAConfig | None = None) -> str:
    """Fine-tune with QLoRA and return the saved adapter path. Requires the `finetune` extra."""
    cfg = config or QLoRAConfig()
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover - only hit without the extra
        raise RuntimeError(
            "LoRA training needs the ML stack: pip install 'caregraph[finetune]' "
            "and run on a CUDA-capable machine."
        ) from exc

    quant = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, quantization_config=quant, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules, task_type="CAUSAL_LM",
    ))

    def format_example(row: dict[str, str]) -> dict[str, str]:
        text = (f"<|system|>\n{row['instruction']}\n<|user|>\n{row['input']}\n"
                f"<|assistant|>\n{row['output']}")
        return {"text": text}

    dataset = load_dataset("json", data_files=data_jsonl, split="train").map(format_example)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.per_device_batch_size,
            learning_rate=cfg.learning_rate,
            max_seq_length=cfg.max_seq_len,
            logging_steps=10,
        ),
    )
    trainer.train()
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    return cfg.output_dir


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="QLoRA fine-tune for problem extraction.")
    parser.add_argument("--data", required=True, help="instruction JSONL path")
    parser.add_argument("--base", default=QLoRAConfig.base_model)
    parser.add_argument("--out", default=QLoRAConfig.output_dir)
    args = parser.parse_args()
    path = train_lora(args.data, QLoRAConfig(base_model=args.base, output_dir=args.out))
    print(f"adapter saved to {path}")
