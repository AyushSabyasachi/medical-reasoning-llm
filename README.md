# Medical Reasoning LLM Fine-Tuning using QLoRA

This project implements a complete fine-tuning and evaluation pipeline for medical reasoning large language models using QLoRA, PEFT, and HuggingFace Transformers.

The project fine-tunes the `Qwen2.5-3B-Instruct` model on the `OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2` dataset and compares:

- Chain-of-Thought (CoT) reasoning fine-tuning
- Direct answer / No-CoT fine-tuning

The repository includes:
- GPU-optimized QLoRA training
- Automatic checkpoint resume support
- CoT reasoning formatting
- Evaluation framework for reasoning quality and hallucination analysis
- Comparative benchmarking between reasoning-enabled and concise-answer models

---

## Features

- QLoRA-based parameter efficient fine-tuning
- PEFT adapter training using LoRA
- CoT vs No-CoT experimental comparison
- HuggingFace + TRL training pipeline
- Automatic checkpoint recovery and resume
- Medical reasoning dataset support
- GPU optimized inference and evaluation
- Hallucination-aware evaluation framework
- Comparative latency and token analysis

---

## Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- TRL (SFTTrainer)
- PEFT
- BitsAndBytes
- QLoRA
- Qwen2.5-3B
- Accelerate
- Datasets

---

## Dataset

Dataset used:

`OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2`

The dataset contains:
- Medical QA conversations
- Reasoning traces
- Assistant explanations
- Multi-turn instruction-following samples

---

## Training Experiments

### Experiment 1 — CoT Fine-Tuning

The assistant is trained to generate explicit reasoning traces:

```text
<think>
reasoning...
</think>

<answer>
final response
</answer>

### Experiment 2 — No-CoT Fine-Tuning

The assistant is trained to generate concise direct answers without explicit reasoning traces.

### Example Training Command
CoT Training
python phase2_train.py \
  --experiment exp1_cot \
  --output_dir ./checkpoints/exp1_cot \
  --batch_size 2 \
  --max_samples 5000

No-CoT Training
python phase2_train.py \
  --experiment exp2_no_cot \
  --output_dir ./checkpoints/exp2_no_cot \
  --batch_size 2 \
  --max_samples 5000

Example Evaluation Command
python phase3_evaluate.py \
    --exp1_dir ./checkpoints/exp1_cot/final_adapter \
    --exp2_dir ./checkpoints/exp2_no_cot/final_adapter \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --output_dir ./evaluation_results \
    --num_eval_samples 200

Evaluation Metrics

The evaluation framework measures:

Exact Match (EM)
ROUGE-L
BERTScore
Hallucination rate
Response latency
Token generation overhead
Reasoning quality comparison

medical-reasoning-llm/
│
├── phase2_train.py
├── phase3_evaluate.py
├── README.md
├── .gitignore
│
├── checkpoints/
├── evaluation_results/
└── reports/

Key Learnings
CoT fine-tuning improves reasoning transparency but increases computational complexity.
Long reasoning traces significantly increase token generation overhead.
Stable QLoRA training requires careful learning rate and gradient stabilization.
No-CoT models are computationally cheaper and more numerically stable.
Future Improvements
Multi-GPU distributed training
RLHF alignment
Gradio/Streamlit inference UI
HuggingFace Spaces deployment
Better hallucination detection
Medical benchmark evaluation
Quantitative reasoning analysis
Disclaimer

This project is intended for research and educational purposes only.

The generated medical responses should not be used as professional medical advice.

Author

Ayush Kumar Singh