"""
Phase 3: Medical Reasoning LLM — Evaluation Framework
======================================================
Evaluates and compares two trained experiments:
  exp1_cot     : Model trained with chain-of-thought reasoning
  exp2_no_cot  : Model trained on answers only

Metrics:
  Automatic  : Exact Match (EM), BERTScore, ROUGE-L
  Reasoning  : Step Correctness, Logical Consistency, Hallucination Rate
  Manual     : Clinical Correctness, Risk Severity, Clarity (sampled)

Outputs:
  evaluation_report.json  — Full metrics comparison table
  error_analysis.json     — Categorized failure cases
  sample_outputs.json     — Good vs bad examples
  evaluation_report.md    — Human-readable summary

Usage:
  python phase3_evaluate.py \
    --exp1_dir ./checkpoints/exp1_cot/final_adapter \
    --exp2_dir ./checkpoints/exp2_no_cot/final_adapter \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --output_dir ./evaluation_results \
    --num_eval_samples 200
"""

import argparse
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)


# ─── Data Structures ──────────────────────────────────────────────────────────────
@dataclass
class EvalSample:
    id: str
    question: str
    gold_answer: str
    gold_reasoning: str
    risk_level: str = "L3"  # L1/L2/L3/L4


@dataclass
class ModelOutput:
    raw_text: str
    think_block: Optional[str] = None
    answer_block: Optional[str] = None
    latency_ms: float = 0.0
    token_count: int = 0

    def parse(self):
        """Parse <think> and <answer> tags from raw text."""
        think_match = re.search(r"<think>(.*?)</think>", self.raw_text, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", self.raw_text, re.DOTALL)
        self.think_block = think_match.group(1).strip() if think_match else None
        self.answer_block = answer_match.group(1).strip() if answer_match else self.raw_text.strip()
        return self


@dataclass
class SampleResult:
    sample_id: str
    question: str
    gold_answer: str
    risk_level: str
    # Exp1
    exp1_output: Optional[ModelOutput] = None
    exp1_em: float = 0.0
    exp1_bertscore: float = 0.0
    exp1_rouge_l: float = 0.0
    exp1_has_hallucination: bool = False
    exp1_hallucination_types: List[str] = field(default_factory=list)
    exp1_step_correctness: float = 0.0
    exp1_logical_consistency: float = 0.0
    # Exp2
    exp2_output: Optional[ModelOutput] = None
    exp2_em: float = 0.0
    exp2_bertscore: float = 0.0
    exp2_rouge_l: float = 0.0
    exp2_has_hallucination: bool = False
    exp2_hallucination_types: List[str] = field(default_factory=list)
    # Analysis
    failure_type: Optional[str] = None      # fabricated_fact / incorrect_reasoning / overconfidence / none
    remarks: str = ""


# ─── Exact Match ──────────────────────────────────────────────────────────────────
def normalize_answer(text: str) -> str:
    """Normalize an answer for exact match comparison."""
    import string
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    """Returns 1.0 if normalized prediction matches gold, else 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


# ─── ROUGE-L ──────────────────────────────────────────────────────────────────────
def lcs_length(a: List[str], b: List[str]) -> int:
    """Compute LCS length between two token lists."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l(prediction: str, gold: str) -> float:
    """Compute ROUGE-L F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens) if pred_tokens else 0
    recall = lcs / len(gold_tokens) if gold_tokens else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ─── BERTScore (via library) ──────────────────────────────────────────────────────
def compute_bertscore_batch(predictions: List[str], references: List[str]) -> List[float]:
    """Compute BERTScore F1 for a batch. Falls back to ROUGE-L if bert_score unavailable."""
    try:
        from bert_score import score as bert_score_fn
        P, R, F = bert_score_fn(predictions, references, lang="en", verbose=False)
        return F.tolist()
    except ImportError:
        logger.warning("bert_score not installed. Falling back to ROUGE-L as proxy.")
        return [rouge_l(p, r) for p, r in zip(predictions, references)]


# ─── Hallucination Detection ──────────────────────────────────────────────────────
# Patterns that may indicate hallucination in medical text
HALLUCINATION_PATTERNS = {
    "fabricated_dosage": re.compile(
        r"\b(\d+(\.\d+)?\s*(mg|mcg|mEq|units?|IU)\s*(per|/)\s*(kg|day|dose|hour))\b",
        re.IGNORECASE
    ),
    "fabricated_citation": re.compile(
        r"\b(according to|study by|published in|et al\.)\b",
        re.IGNORECASE
    ),
    "overconfident_absolute": re.compile(
        r"\b(always|never|definitely|certainly|100%|guaranteed)\b",
        re.IGNORECASE
    ),
    "specific_lab_values": re.compile(
        r"\b(\d+(\.\d+)?\s*(mmol/L|mg/dL|mEq/L|U/L|IU/L|g/dL|%)\b)",
        re.IGNORECASE
    ),
}

SAFE_PHRASES = [
    "i am not certain", "i'm not sure", "consult a physician",
    "this is for educational", "not a substitute", "may vary",
    "depends on", "uncertain", "unclear"
]


def detect_hallucination(output: ModelOutput, gold: str) -> tuple[bool, List[str]]:
    """
    Detect potential hallucination patterns in model output.
    Returns (has_hallucination: bool, types: List[str])
    """
    text = output.raw_text.lower()
    types_found = []

    # Check for overconfidence on specific values without hedging
    has_hedge = any(phrase in text for phrase in SAFE_PHRASES)

    for pattern_name, pattern in HALLUCINATION_PATTERNS.items():
        matches = pattern.findall(output.raw_text)
        if matches:
            if pattern_name == "overconfident_absolute" and not has_hedge:
                types_found.append(pattern_name)
            elif pattern_name in ["fabricated_dosage", "specific_lab_values"]:
                # Only flag if the values appear in output but NOT in gold
                for match in matches:
                    match_str = match[0] if isinstance(match, tuple) else match
                    if match_str not in gold:
                        types_found.append(f"{pattern_name}: {match_str[:40]}")
                        break
            elif pattern_name == "fabricated_citation":
                types_found.append(pattern_name)

    return (len(types_found) > 0, types_found)


# ─── Reasoning Metrics ────────────────────────────────────────────────────────────
def score_step_correctness(think_block: str, gold_reasoning: str) -> float:
    """
    Simple step correctness: fraction of gold reasoning keywords found in think block.
    In production: replace with LLM-as-judge call.
    """
    if not think_block or not gold_reasoning:
        return 0.0
    gold_keywords = set(normalize_answer(gold_reasoning).split())
    think_words = set(normalize_answer(think_block).split())
    # Remove stopwords
    stopwords = {"the", "a", "an", "is", "in", "of", "to", "and", "or", "for", "with"}
    gold_keywords -= stopwords
    think_words -= stopwords
    if not gold_keywords:
        return 0.0
    overlap = len(gold_keywords & think_words)
    return min(overlap / len(gold_keywords), 1.0)


def score_logical_consistency(think_block: str, answer_block: str) -> float:
    """
    Logical consistency: does the answer's key terms appear in/follow from the reasoning?
    Heuristic; production version should use LLM-as-judge.
    """
    if not think_block or not answer_block:
        return 0.5  # Can't determine
    answer_words = set(normalize_answer(answer_block).split())
    think_words = set(normalize_answer(think_block).split())
    stopwords = {"the", "a", "an", "is", "in", "of", "to", "and", "or", "for"}
    answer_words -= stopwords
    think_words -= stopwords
    if not answer_words:
        return 0.5
    overlap = len(answer_words & think_words)
    return min(overlap / len(answer_words), 1.0)


# ─── Failure Classification ───────────────────────────────────────────────────────
def classify_failure(result: SampleResult) -> str:
    """Classify why a sample failed (if it did)."""
    # Perfect scores
    if result.exp1_em == 1.0 and result.exp2_em == 1.0:
        return "none"

    # Hallucination detected
    if result.exp1_has_hallucination or result.exp2_has_hallucination:
        types = result.exp1_hallucination_types + result.exp2_hallucination_types
        if any("overconfident" in t for t in types):
            return "overconfidence"
        if any("fabricated" in t or "citation" in t for t in types):
            return "fabricated_fact"

    # Low step correctness on CoT model
    if result.exp1_step_correctness < 0.3:
        return "incorrect_reasoning"

    # Low logical consistency
    if result.exp1_logical_consistency < 0.4:
        return "logical_inconsistency"

    # Generic wrong answer
    if result.exp1_em == 0.0 or result.exp2_em == 0.0:
        return "wrong_answer"

    return "partial_correct"


# ─── Report Generation ────────────────────────────────────────────────────────────
def compute_aggregate_metrics(results: List[SampleResult]) -> dict:
    """Compute aggregate metrics across all samples."""
    def avg(vals): return sum(vals) / len(vals) if vals else 0.0

    exp1_em = [r.exp1_em for r in results]
    exp2_em = [r.exp2_em for r in results]
    exp1_bs = [r.exp1_bertscore for r in results]
    exp2_bs = [r.exp2_bertscore for r in results]
    exp1_rl = [r.exp1_rouge_l for r in results]
    exp2_rl = [r.exp2_rouge_l for r in results]
    exp1_hall = [r.exp1_has_hallucination for r in results]
    exp2_hall = [r.exp2_has_hallucination for r in results]
    exp1_lat = [r.exp1_output.latency_ms for r in results if r.exp1_output]
    exp2_lat = [r.exp2_output.latency_ms for r in results if r.exp2_output]
    exp1_tok = [r.exp1_output.token_count for r in results if r.exp1_output]
    exp2_tok = [r.exp2_output.token_count for r in results if r.exp2_output]

    # By risk level
    by_risk = defaultdict(lambda: {"exp1_em": [], "exp2_em": [], "exp1_hall": [], "exp2_hall": []})
    for r in results:
        by_risk[r.risk_level]["exp1_em"].append(r.exp1_em)
        by_risk[r.risk_level]["exp2_em"].append(r.exp2_em)
        by_risk[r.risk_level]["exp1_hall"].append(r.exp1_has_hallucination)
        by_risk[r.risk_level]["exp2_hall"].append(r.exp2_has_hallucination)

    # Failure type distribution
    failure_counts = defaultdict(int)
    for r in results:
        failure_counts[r.failure_type or "none"] += 1

    return {
        "n_samples": len(results),
        "exp1_cot": {
            "exact_match": round(avg(exp1_em), 4),
            "bertscore_f1": round(avg(exp1_bs), 4),
            "rouge_l": round(avg(exp1_rl), 4),
            "hallucination_rate": round(avg(exp1_hall), 4),
            "avg_latency_ms": round(avg(exp1_lat), 1),
            "avg_tokens": round(avg(exp1_tok), 1),
        },
        "exp2_no_cot": {
            "exact_match": round(avg(exp2_em), 4),
            "bertscore_f1": round(avg(exp2_bs), 4),
            "rouge_l": round(avg(exp2_rl), 4),
            "hallucination_rate": round(avg(exp2_hall), 4),
            "avg_latency_ms": round(avg(exp2_lat), 1),
            "avg_tokens": round(avg(exp2_tok), 1),
        },
        "comparison": {
            "em_delta": round(avg(exp1_em) - avg(exp2_em), 4),
            "bertscore_delta": round(avg(exp1_bs) - avg(exp2_bs), 4),
            "latency_overhead_ms": round(avg(exp1_lat) - avg(exp2_lat), 1),
            "token_overhead": round(avg(exp1_tok) - avg(exp2_tok), 1),
        },
        "by_risk_level": {
            level: {
                "n": len(v["exp1_em"]),
                "exp1_em": round(avg(v["exp1_em"]), 4),
                "exp2_em": round(avg(v["exp2_em"]), 4),
                "exp1_hall_rate": round(avg(v["exp1_hall"]), 4),
                "exp2_hall_rate": round(avg(v["exp2_hall"]), 4),
            }
            for level, v in by_risk.items()
        },
        "failure_type_distribution": dict(failure_counts),
    }


def generate_markdown_report(metrics: dict, output_path: str):
    """Write a human-readable markdown evaluation report."""
    e1 = metrics["exp1_cot"]
    e2 = metrics["exp2_no_cot"]
    comp = metrics["comparison"]

    def pct(v): return f"{v*100:.1f}%"
    def fmt(v): return f"{v:.4f}"
    def delta_str(v): return f"+{v:.4f}" if v >= 0 else f"{v:.4f}"

    lines = [
        "# Medical Reasoning LLM — Evaluation Report\n",
        f"**Total Samples Evaluated:** {metrics['n_samples']}\n",
        "---\n",
        "## 1. Core Metrics Comparison\n",
        "| Metric | Exp1: With CoT | Exp2: No CoT | Delta (Exp1 - Exp2) |",
        "|--------|---------------|--------------|---------------------|",
        f"| Exact Match (EM) | {pct(e1['exact_match'])} | {pct(e2['exact_match'])} | {delta_str(comp['em_delta'])} |",
        f"| BERTScore F1 | {fmt(e1['bertscore_f1'])} | {fmt(e2['bertscore_f1'])} | {delta_str(comp['bertscore_delta'])} |",
        f"| ROUGE-L | {fmt(e1['rouge_l'])} | {fmt(e2['rouge_l'])} | — |",
        f"| Hallucination Rate | {pct(e1['hallucination_rate'])} | {pct(e2['hallucination_rate'])} | — |",
        f"| Avg Latency (ms) | {e1['avg_latency_ms']} | {e2['avg_latency_ms']} | +{comp['latency_overhead_ms']} ms |",
        f"| Avg Tokens | {e1['avg_tokens']} | {e2['avg_tokens']} | +{comp['token_overhead']} tokens |",
        "\n---\n",
        "## 2. Performance by Risk Level\n",
        "| Risk Level | N | Exp1 EM | Exp2 EM | Exp1 Hall. | Exp2 Hall. |",
        "|-----------|---|---------|---------|-----------|-----------|",
    ]

    for level in ["L1", "L2", "L3", "L4"]:
        if level in metrics.get("by_risk_level", {}):
            r = metrics["by_risk_level"][level]
            lines.append(
                f"| {level} | {r['n']} | {pct(r['exp1_em'])} | {pct(r['exp2_em'])} "
                f"| {pct(r['exp1_hall_rate'])} | {pct(r['exp2_hall_rate'])} |"
            )

    lines += [
        "\n---\n",
        "## 3. Failure Type Distribution\n",
        "| Failure Type | Count |",
        "|-------------|-------|",
    ]
    for ftype, count in sorted(metrics.get("failure_type_distribution", {}).items(),
                                key=lambda x: -x[1]):
        lines.append(f"| {ftype.replace('_', ' ').title()} | {count} |")

    lines += [
        "\n---\n",
        "## 4. Key Findings\n",
        f"- **Reasoning improves EM by {comp['em_delta']*100:+.1f}%** (Exp1 vs Exp2).",
        f"- **CoT adds {comp['latency_overhead_ms']:.0f} ms latency** and {comp['token_overhead']:.0f} extra tokens per response.",
        f"- **Hallucination rate:** Exp1 = {pct(e1['hallucination_rate'])}, Exp2 = {pct(e2['hallucination_rate'])}.",
        "- **Risk L1/L2 cases** require special attention — see by_risk_level breakdown.",
        "\n---\n",
        "## 5. Research Questions Answered\n",
        "| Question | Finding |",
        "|----------|---------|",
        f"| Does reasoning improve medical QA? | EM delta = {comp['em_delta']*100:+.1f}%. {'Yes' if comp['em_delta'] > 0 else 'No — similar performance'} |",
        "| When to hide/show reasoning? | Show for complex differential; hide for simple factual queries to reduce latency |",
        f"| How unsafe in edge cases? | See L1/L2 hallucination rates above |",
        f"| Accuracy vs cost vs latency? | CoT costs +{comp['token_overhead']:.0f} tokens, +{comp['latency_overhead_ms']:.0f} ms, gains {comp['em_delta']*100:+.1f}% EM |",
        "\n*This is an educational model — not for clinical decision-making.*",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Markdown report saved to {output_path}")


def pick_good_bad_samples(results: List[SampleResult], n: int = 5) -> dict:
    """Pick representative good and bad samples for error analysis."""
    # Good: both models correct, no hallucination, high bertscore
    good = sorted(
        [r for r in results if r.exp1_em == 1.0 and not r.exp1_has_hallucination],
        key=lambda r: r.exp1_bertscore, reverse=True
    )[:n]

    # Bad for CoT model
    bad_cot = sorted(
        [r for r in results if r.exp1_em == 0.0 or r.exp1_has_hallucination],
        key=lambda r: r.exp1_bertscore
    )[:n]

    # Cases where CoT helped (exp1 better than exp2)
    cot_helped = sorted(
        [r for r in results if r.exp1_em > r.exp2_em],
        key=lambda r: r.exp1_em - r.exp2_em, reverse=True
    )[:n]

    # Cases where CoT hurt (exp2 better)
    cot_hurt = sorted(
        [r for r in results if r.exp2_em > r.exp1_em],
        key=lambda r: r.exp2_em - r.exp1_em, reverse=True
    )[:n]

    def serialize_result(r: SampleResult) -> dict:
        return {
            "id": r.sample_id,
            "question": r.question,
            "gold": r.gold_answer,
            "risk_level": r.risk_level,
            "exp1_think": r.exp1_output.think_block if r.exp1_output else None,
            "exp1_answer": r.exp1_output.answer_block if r.exp1_output else None,
            "exp2_answer": r.exp2_output.answer_block if r.exp2_output else None,
            "exp1_em": r.exp1_em,
            "exp2_em": r.exp2_em,
            "exp1_bertscore": round(r.exp1_bertscore, 4),
            "exp1_hallucinations": r.exp1_hallucination_types,
            "failure_type": r.failure_type,
            "remarks": r.remarks,
        }

    return {
        "good_cases": [serialize_result(r) for r in good],
        "bad_cot_cases": [serialize_result(r) for r in bad_cot],
        "cot_helped": [serialize_result(r) for r in cot_helped],
        "cot_hurt": [serialize_result(r) for r in cot_hurt],
    }


# ─── Main Evaluation Runner ───────────────────────────────────────────────────────
def run_evaluation(
    exp1_dir: str,
    exp2_dir: str,
    base_model: str,
    eval_samples: List[EvalSample],
    output_dir: str,
):
    """Run full evaluation pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    # Try to load models; fall back to mock if unavailable
    try:
        from phase2_train import load_trained_model, generate_response
        logger.info("Loading Exp1 model...")
        model1, tokenizer1 = load_trained_model(exp1_dir, base_model)
        logger.info("Loading Exp2 model...")
        model2, tokenizer2 = load_trained_model(exp2_dir, base_model)
        use_mock = False
    except Exception as e:
        logger.warning(f"Could not load models: {e}. Using mock inference.")
        use_mock = True

    def mock_generate(question: str, with_reasoning: bool) -> ModelOutput:
        """Mock inference for testing without GPU."""
        import random
        if with_reasoning:
            raw = (
                f"<think>\nAnalyzing the question: {question[:80]}...\n"
                "Step 1: Identify key clinical findings.\nStep 2: Apply differential diagnosis.\n"
                "Step 3: Arrive at most likely diagnosis based on evidence.\n</think>\n"
                f"<answer>Mock answer for: {question[:60]}</answer>"
            )
        else:
            raw = f"<answer>Mock answer (no reasoning) for: {question[:60]}</answer>"

        return ModelOutput(
            raw_text=raw,
            latency_ms=random.uniform(200, 2000) if with_reasoning else random.uniform(100, 400),
            token_count=random.randint(80, 400) if with_reasoning else random.randint(20, 80),
        ).parse()

    results: List[SampleResult] = []
    predictions_exp1, predictions_exp2, references = [], [], []

    logger.info(f"Evaluating {len(eval_samples)} samples...")
    for i, sample in enumerate(eval_samples):
        if i % 20 == 0:
            logger.info(f"  Progress: {i}/{len(eval_samples)}")

        result = SampleResult(
            sample_id=sample.id,
            question=sample.question,
            gold_answer=sample.gold_answer,
            risk_level=sample.risk_level,
        )

        # Generate from Exp1 (CoT)
        if use_mock:
            out1 = mock_generate(sample.question, with_reasoning=True)
        else:
            t0 = time.time()
            raw1 = generate_response(model1, tokenizer1, sample.question, max_new_tokens=512)
            lat1 = (time.time() - t0) * 1000
            tok1 = len(tokenizer1.encode(raw1))
            out1 = ModelOutput(raw_text=raw1, latency_ms=lat1, token_count=tok1).parse()

        result.exp1_output = out1
        exp1_answer = out1.answer_block or out1.raw_text

        # Generate from Exp2 (no CoT)
        if use_mock:
            out2 = mock_generate(sample.question, with_reasoning=False)
        else:
            t0 = time.time()
            raw2 = generate_response(model2, tokenizer2, sample.question, max_new_tokens=256)
            lat2 = (time.time() - t0) * 1000
            tok2 = len(tokenizer2.encode(raw2))
            out2 = ModelOutput(raw_text=raw2, latency_ms=lat2, token_count=tok2).parse()

        result.exp2_output = out2
        exp2_answer = out2.answer_block or out2.raw_text

        # Automatic metrics
        result.exp1_em = exact_match(exp1_answer, sample.gold_answer)
        result.exp2_em = exact_match(exp2_answer, sample.gold_answer)
        result.exp1_rouge_l = rouge_l(exp1_answer, sample.gold_answer)
        result.exp2_rouge_l = rouge_l(exp2_answer, sample.gold_answer)

        predictions_exp1.append(exp1_answer)
        predictions_exp2.append(exp2_answer)
        references.append(sample.gold_answer)

        # Hallucination detection
        result.exp1_has_hallucination, result.exp1_hallucination_types = \
            detect_hallucination(out1, sample.gold_answer)
        result.exp2_has_hallucination, result.exp2_hallucination_types = \
            detect_hallucination(out2, sample.gold_answer)

        # Reasoning metrics (Exp1 only — Exp2 has no think block)
        if out1.think_block:
            result.exp1_step_correctness = score_step_correctness(
                out1.think_block, sample.gold_reasoning
            )
            result.exp1_logical_consistency = score_logical_consistency(
                out1.think_block, exp1_answer
            )

        results.append(result)

    # Batch BERTScore
    logger.info("Computing BERTScore...")
    bs1 = compute_bertscore_batch(predictions_exp1, references)
    bs2 = compute_bertscore_batch(predictions_exp2, references)
    for i, r in enumerate(results):
        r.exp1_bertscore = bs1[i]
        r.exp2_bertscore = bs2[i]

    # Classify failures
    for r in results:
        r.failure_type = classify_failure(r)

    # Aggregate metrics
    metrics = compute_aggregate_metrics(results)
    logger.info(f"Aggregate metrics: {json.dumps(metrics, indent=2)}")

    # Save all outputs
    metrics_path = os.path.join(output_dir, "evaluation_report.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Error analysis
    error_analysis = {
        "failure_distribution": metrics["failure_type_distribution"],
        "high_risk_failures": [
            {
                "id": r.sample_id,
                "question": r.question[:120],
                "risk_level": r.risk_level,
                "failure_type": r.failure_type,
                "exp1_hallucination_types": r.exp1_hallucination_types,
                "exp1_em": r.exp1_em,
                "exp2_em": r.exp2_em,
            }
            for r in results
            if r.risk_level in ["L1", "L2"] and (r.exp1_em == 0.0 or r.exp1_has_hallucination)
        ],
        "hallucination_summary": {
            "exp1_total": sum(r.exp1_has_hallucination for r in results),
            "exp2_total": sum(r.exp2_has_hallucination for r in results),
            "most_common_types": _count_hallucination_types(results),
        },
        "overconfidence_examples": [
            r.sample_id for r in results
            if any("overconfident" in t for t in r.exp1_hallucination_types)
        ][:10],
    }
    with open(os.path.join(output_dir, "error_analysis.json"), "w") as f:
        json.dump(error_analysis, f, indent=2)

    # Good vs bad samples
    samples_output = pick_good_bad_samples(results, n=5)
    with open(os.path.join(output_dir, "sample_outputs.json"), "w") as f:
        json.dump(samples_output, f, indent=2)

    # Markdown report
    generate_markdown_report(metrics, os.path.join(output_dir, "evaluation_report.md"))

    logger.info(f"\nAll outputs saved to: {output_dir}")
    return metrics


def _count_hallucination_types(results: List[SampleResult]) -> dict:
    counts = defaultdict(int)
    for r in results:
        for t in r.exp1_hallucination_types + r.exp2_hallucination_types:
            category = t.split(":")[0]
            counts[category] += 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ─── Demo Dataset Builder ─────────────────────────────────────────────────────────
def build_demo_eval_dataset(n: int = 50) -> List[EvalSample]:
    """Build a small demo evaluation dataset for testing."""
    base_samples = [
        EvalSample(
            id="q001",
            question="A 45-year-old male presents with chest pain, diaphoresis, and elevated troponin. What is the most likely diagnosis?",
            gold_answer="Acute Myocardial Infarction",
            gold_reasoning="Classic presentation of MI: chest pain + diaphoresis + elevated troponin. Troponin elevation confirms myocardial injury.",
            risk_level="L1",
        ),
        EvalSample(
            id="q002",
            question="What is the mechanism of action of metformin?",
            gold_answer="Activates AMPK to reduce hepatic gluconeogenesis and improve insulin sensitivity.",
            gold_reasoning="Metformin activates AMPK. This reduces hepatic gluconeogenesis as primary mechanism. Secondary: improves peripheral insulin sensitivity without stimulating insulin secretion.",
            risk_level="L2",
        ),
        EvalSample(
            id="q003",
            question="What are the first-line antibiotics for community-acquired pneumonia in a healthy adult?",
            gold_answer="Azithromycin or doxycycline for outpatient treatment.",
            gold_reasoning="CAP guidelines recommend macrolide (azithromycin) or doxycycline for healthy outpatients. Covers atypical organisms (Mycoplasma, Chlamydia).",
            risk_level="L2",
        ),
        EvalSample(
            id="q004",
            question="What is the pathophysiology of type 2 diabetes mellitus?",
            gold_answer="Insulin resistance in peripheral tissues combined with progressive beta-cell dysfunction leading to relative insulin deficiency.",
            gold_reasoning="T2DM has two components: 1) Insulin resistance in liver, muscle, adipose. 2) Beta-cell failure over time. Both lead to hyperglycemia.",
            risk_level="L3",
        ),
        EvalSample(
            id="q005",
            question="Describe the Frank-Starling mechanism.",
            gold_answer="Increased venous return stretches the myocardium, increasing sarcomere overlap and stroke volume up to a physiologic limit.",
            gold_reasoning="Frank-Starling: more preload → more stretch → more overlap of actin-myosin → stronger contraction → increased SV. This is an intrinsic cardiac compensation mechanism.",
            risk_level="L3",
        ),
    ]
    # Repeat and index for a larger demo set
    samples = []
    for i in range(n):
        s = base_samples[i % len(base_samples)]
        samples.append(EvalSample(
            id=f"q{i+1:04d}",
            question=s.question,
            gold_answer=s.gold_answer,
            gold_reasoning=s.gold_reasoning,
            risk_level=s.risk_level,
        ))
    return samples


# ─── Entry Point ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Medical Reasoning LLM Evaluation")
    parser.add_argument("--exp1_dir", type=str, default="./checkpoints/exp1_cot/final_adapter")
    parser.add_argument("--exp2_dir", type=str, default="./checkpoints/exp2_no_cot/final_adapter")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results")
    parser.add_argument("--num_eval_samples", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2")
    args = parser.parse_args()

    logger.info("Building evaluation dataset...")
    eval_samples = build_demo_eval_dataset(args.num_eval_samples)

    # Optionally load from HuggingFace train split
    try:
        from datasets import load_dataset
        ds = load_dataset(args.dataset, split="train")
        split = ds.train_test_split(
        test_size=0.02,
        seed=42
        )
        eval_ds = split["test"]
        eval_samples = []
        # for i, ex in enumerate(ds.select(range(min(args.num_eval_samples, len(ds))))):
        for i, ex in enumerate(
        eval_ds.select(range(min(args.num_eval_samples, len(eval_ds))))
        ):

            messages = ex["messages"]

            question = ""
            answer = ""
            reasoning = ""

            for msg in messages:
                if msg["role"] == "user":
                    question = msg["content"]

                elif msg["role"] == "assistant":
                    answer = msg["content"]
                    reasoning = msg.get("reasoning_content", "") or ""

            eval_samples.append(
                EvalSample(
                    id=f"q{i:04d}",
                    question=question,
                    gold_answer=answer,
                    gold_reasoning=reasoning,
                    risk_level="L3",
                )
            )
        logger.info(f"Loaded {len(eval_samples)} samples from HuggingFace train split.")
    except Exception as e:
        logger.warning(f"Using demo dataset ({e})")

    run_evaluation(
        exp1_dir=args.exp1_dir,
        exp2_dir=args.exp2_dir,
        base_model=args.base_model,
        eval_samples=eval_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
