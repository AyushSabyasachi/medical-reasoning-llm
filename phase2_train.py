"""
Phase 2: Medical Reasoning LLM — Training Pipeline
====================================================
Dataset  : OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2
Base Model: Qwen/Qwen2.5-3B-Instruct
Method   : SFT + QLoRA (4-bit)
Experiments:
  - exp1_cot  : Train on full chain-of-thought reasoning traces
  - exp2_no_cot: Train only on final answers (no reasoning)

Usage:
  python phase2_train.py --experiment exp1_cot   --output_dir ./checkpoints/exp1
  python phase2_train.py --experiment exp2_no_cot --output_dir ./checkpoints/exp2

Requirements:
  pip install transformers peft trl datasets bitsandbytes accelerate wandb torch
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from transformers import BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

# ─── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    # Model
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"

    # Dataset
    dataset_name: str = "OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2"
    dataset_split: str = "train"
    val_split_ratio: float = 0.05  # 5% validation
    max_train_samples: Optional[int] = None  # Set to e.g. 5000 for quick testing

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training
    num_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2   # effective batch = 16
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 50

    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True

    # Output
    output_dir: str = "./checkpoints/exp1_cot"
    experiment: str = "exp1_cot"  # "exp1_cot" | "exp2_no_cot"

    # Misc
    seed: int = 42
    use_wandb: bool = False
    wandb_project: str = "medical-reasoning-llm"


# ─── System Prompt ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a cautious medical reasoning assistant. "
    "For every query, you must:\n"
    "  1. Think step-by-step inside <think>...</think> tags.\n"
    "  2. Provide a final, concise answer inside <answer>...</answer> tags.\n"
    "  3. Never fabricate drug dosages, lab values, or clinical guidelines.\n"
    "  4. Express uncertainty explicitly when unsure.\n"
    "  5. This is for educational purposes only — not clinical decision support."
)


# ─── Data Preprocessing ───────────────────────────────────────────────────────────
def build_cot_output(reasoning: str, answer: str, max_reasoning_tokens: int = 800) -> str:
    """Build CoT output: <think>reasoning</think><answer>answer</answer>"""
    # Truncate reasoning to avoid extremely long sequences
    words = reasoning.split()
    if len(words) > max_reasoning_tokens:
        reasoning = " ".join(words[:max_reasoning_tokens]) + "..."
    return f"<think>\n{reasoning.strip()}\n</think>\n<answer>{answer.strip()}</answer>"


def build_answer_only_output(answer: str) -> str:
    """Build answer-only output: <answer>answer</answer>"""
    return f"<answer>{answer.strip()}</answer>"


def format_example(example: dict, tokenizer, experiment: str) -> dict:
    """
    Format a dataset example into the Qwen2.5 chat template.
    Returns a dict with a 'text' key (full formatted string for SFT).
    """
    question = example.get("question", example.get("input", ""))
    reasoning = example.get("reasoning", example.get("chain_of_thought", ""))
    answer = example.get("answer", example.get("output", ""))

    # Build assistant response based on experiment
    if experiment == "exp1_cot" and reasoning:
        assistant_response = build_cot_output(reasoning, answer)
    else:
        assistant_response = build_answer_only_output(answer)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_response},
    ]

    # Apply Qwen2.5 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def load_and_preprocess(config: TrainingConfig, tokenizer):
    """Load dataset and apply preprocessing."""
    logger.info(f"Loading dataset: {config.dataset_name}")
    try:
        dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    except Exception as e:
        logger.warning(f"Could not load {config.dataset_name}: {e}")
        logger.info("Generating synthetic demo dataset for testing...")
        dataset = _create_demo_dataset()

    if config.max_train_samples:
        dataset = dataset.shuffle(seed=config.seed).select(
            range(min(config.max_train_samples, len(dataset)))
        )
        logger.info(f"Subsampled to {len(dataset)} examples.")

    # Train/val split
    split = dataset.train_test_split(
        test_size=config.val_split_ratio, seed=config.seed
    )
    train_ds = split["train"]
    val_ds = split["test"]

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Format
    # fmt_fn = lambda ex: format_example(ex, tokenizer, config.experiment)
    # train_ds = train_ds.map(fmt_fn, remove_columns=train_ds.column_names, num_proc=4)
    # val_ds = val_ds.map(fmt_fn, remove_columns=val_ds.column_names, num_proc=4)

    # Filter by max_seq_length
    def length_filter(examples):
        batch_size = len(examples["messages"])
        texts = []

        for i in range(batch_size):
            ex = {
                key: examples[key][i]
                for key in examples
            }

            text = formatting_func(ex, tokenizer)
            texts.append(text)
            tokenized = tokenizer(texts, truncation=False)

        return [
        len(ids) <= config.max_seq_length
        for ids in tokenized["input_ids"]
        ]
    logger.info(f"FINAL TRAIN SIZE: {len(train_ds)}")
    logger.info(f"FINAL VAL SIZE: {len(val_ds)}")
    # train_ds = train_ds.filter(length_filter, batched=True, batch_size=1000)
    # val_ds = val_ds.filter(length_filter, batched=True, batch_size=1000)

    logger.info(f"After length filter — Train: {len(train_ds)} | Val: {len(val_ds)}")

    return train_ds, val_ds

        


def _create_demo_dataset():
    """Fallback synthetic dataset for offline testing."""
    from datasets import Dataset
    samples = [
        {
            "question": "A 45-year-old male presents with chest pain, diaphoresis, and elevated troponin. What is the most likely diagnosis?",
            "reasoning": "The patient has classic symptoms of acute myocardial infarction: chest pain + diaphoresis + elevated troponin (cardiac biomarker). The combination is pathognomonic for MI. Need to differentiate STEMI vs NSTEMI based on ECG.",
            "answer": "Acute Myocardial Infarction (STEMI or NSTEMI). Requires urgent ECG and cardiology consultation.",
        },
        {
            "question": "What is the mechanism of action of metformin?",
            "reasoning": "Metformin is a biguanide. Primary mechanism: activates AMPK (AMP-activated protein kinase) which reduces hepatic gluconeogenesis. Secondary: improves peripheral insulin sensitivity. Does NOT increase insulin secretion (unlike sulfonylureas).",
            "answer": "Metformin activates AMPK, primarily reducing hepatic gluconeogenesis. It also improves peripheral insulin sensitivity without stimulating insulin secretion.",
        },
        {
            "question": "What are the first-line antibiotics for community-acquired pneumonia (CAP) in an otherwise healthy adult?",
            "reasoning": "CAP guidelines (ATS/IDSA): For outpatient treatment in otherwise healthy adults, atypical pathogens (Mycoplasma, Chlamydia) must be covered. Macrolide (azithromycin) or doxycycline is first-line. If co-morbidities or recent antibiotic use: respiratory fluoroquinolone or beta-lactam + macrolide.",
            "answer": "Azithromycin (500 mg day 1, then 250 mg days 2-5) or doxycycline (100 mg twice daily x 5 days) for healthy outpatient CAP.",
        },
    ] * 500  # repeat to create a usable dataset size
    return Dataset.from_list(samples)


# ─── Model Setup ─────────────────────────────────────────────────────────────────
def load_model_and_tokenizer(config: TrainingConfig):
    """Load the base model with 4-bit quantization and apply LoRA."""
    logger.info(f"Loading tokenizer: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # BitsAndBytes 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )

    logger.info(f"Loading model: {config.base_model} (4-bit QLoRA)")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config if config.use_4bit else None,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Prepare for kbit training
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ─── Training ─────────────────────────────────────────────────────────────────────
def build_training_args(config: TrainingConfig) -> TrainingArguments:
    report_to = "wandb" if config.use_wandb else "none"
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_grad_norm=1.0,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=False,
        bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
        gradient_checkpointing=True,
        report_to=report_to,
        run_name=f"medical-llm-{config.experiment}",
        seed=config.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["labels"],
    )


def train(config: TrainingConfig):
    """Main training loop."""
    os.makedirs(config.output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(config.output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2)
    logger.info(f"Config saved to {config_path}")

    if config.use_wandb:
        import wandb
        wandb.init(project=config.wandb_project, name=f"medical-llm-{config.experiment}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(config)
    

    # Load data
    train_ds, val_ds = load_and_preprocess(config, tokenizer)

    # Training args
    training_args = build_training_args(config)

    # Determine response template for completion-only loss
    # For exp1_cot: train on full assistant output
    # For exp2_no_cot: same, but output is shorter
    response_template = "<|im_start|>assistant\n"
    # collator = DataCollatorForCompletionOnlyLM(
    #     response_template=response_template,
    #     # # # # # # tokenizer=tokenizer,
    # )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=lambda ex: formatting_func(ex, tokenizer),
        args=SFTConfig(
            output_dir=config.output_dir,
            # max_seq_length=config.max_seq_length,
            # # # # # # # .max_seq_length,
            # dataset_text_field="text",
            **{k: v for k, v in training_args.to_dict().items()
               if k not in ["output_dir", "max_seq_length"]}
        ),
    )

    logger.info(f"Starting training: {config.experiment}")
    start_time = time.time()
    # trainer.train()
    checkpoint = get_last_checkpoint(config.output_dir)

    if checkpoint:
        logger.info(f"Resuming from checkpoint: {checkpoint}")

    trainer.train(resume_from_checkpoint=checkpoint)
    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed/60:.1f} minutes.")

    # Save final adapter
    adapter_path = os.path.join(config.output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"Adapter saved to {adapter_path}")

    # Save experiment log
    log = {
        "experiment": config.experiment,
        "base_model": config.base_model,
        "dataset": config.dataset_name,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "epochs": config.num_epochs,
        "effective_batch_size": config.per_device_train_batch_size * config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "max_seq_length": config.max_seq_length,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "training_time_minutes": round(elapsed / 60, 2),
        "final_eval_loss": trainer.state.best_metric,
    }
    with open(os.path.join(config.output_dir, "experiment_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    logger.info(f"Experiment log: {log}")

    return trainer


# ─── Inference Helper ─────────────────────────────────────────────────────────────
def load_trained_model(checkpoint_dir: str, base_model: str):
    """Load a trained adapter for inference."""
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", 
        # torch_dtype=torch.float16, 
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    """Generate a response from the fine-tuned model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # Greedy for deterministic eval
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

def formatting_func(example, tokenizer):
        messages = example["messages"]

        converted_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Include reasoning for assistant
            if role == "assistant":
                reasoning = msg.get("reasoning_content")

                if reasoning:
                    content = (
                        f"<think>\n{reasoning.strip()}\n</think>\n"
                        f"<answer>\n{content.strip()}\n</answer>"
                    )

            converted_messages.append({
                "role": role,
                "content": content
            })

        text = tokenizer.apply_chat_template(
            converted_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=2048,
        )

        return tokenizer.decode(tokens["input_ids"], skip_special_tokens=False)

        # return text



# ─── Entry Point ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Medical Reasoning LLM Training")
    parser.add_argument(
        "--experiment", type=str, choices=["exp1_cot", "exp2_no_cot"],
        default="exp1_cot", help="Experiment to run"
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument('--subset_size', type=int, default=0, help='Number of samples to use for training. 0 means use all.')
    args = parser.parse_args()

    config = TrainingConfig(
        experiment=args.experiment,
        base_model=args.base_model,
        output_dir=args.output_dir or f"./checkpoints/{args.experiment}",
        num_epochs=args.epochs,
        max_train_samples=args.max_samples,
        lora_r=args.lora_r,
        learning_rate=args.lr,
        use_wandb=args.use_wandb,
        per_device_train_batch_size=args.batch_size,
    )

    logger.info("=" * 60)
    logger.info(f"EXPERIMENT: {config.experiment.upper()}")
    logger.info(f"Base model: {config.base_model}")
    logger.info(f"Output dir: {config.output_dir}")
    logger.info("=" * 60)

    train(config)


if __name__ == "__main__":
    main()
