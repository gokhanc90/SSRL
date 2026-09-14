import os
import re
import gc
import glob
import argparse

import torch
import numpy as np
import textstat
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel
from easse.sari import corpus_sari
import evaluate
import pandas as pd

# ==========================================
# 1. Configuration
# ==========================================
# Base models keyed by the label used in run names (llama1b_grpo_sari -> "llama1b").
# NOTE: set these to THIS machine's base-model locations (same as training MODELS).
BASE_MODELS = {
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
}

RUNS_DIR = os.environ.get("RUNS_DIR", "runs")            # where the *-Final adapters live
OUT_DIR = os.environ.get("OUT_DIR", "System_Output")     # where system outputs are written
SPLIT = os.environ.get("SPLIT", "test")                  # 'test' for final eval


def clean_output(text):
    patterns = [
        r"^Sure, here is (the )?simplified sentence:?\s*",
        r"^Here is the simplification:?\s*",
        r"^The simplified text is:?\s*",
        r"^Simplify:?\s*",
    ]
    cleaned = text
    for p in patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip().strip('"').strip("'")
    if "\n" in cleaned:
        cleaned = cleaned.split("\n")[0]
    return cleaned.strip()


# ==========================================
# 2. ASSET data (machine-independent resolution)
# ==========================================
ASSET_DIR_CANDIDATES = [
    "asset/dataset",
    "../../asset/dataset",
]


def resolve_asset_dir():
    candidates = list(ASSET_DIR_CANDIDATES)
    try:
        import easse
        candidates.append(os.path.join(
            os.path.dirname(easse.__file__),
            "resources", "data", "test_sets", "asset"))
    except Exception:
        pass
    for d in candidates:
        if os.path.exists(os.path.join(d, f"asset.{SPLIT}.orig")):
            return d
    return candidates[0]


def load_asset_raw():
    """Read ASSET sources + 10 references once (model-independent)."""
    asset_dir = resolve_asset_dir()
    print(f"Using ASSET data dir: {asset_dir}")
    with open(os.path.join(asset_dir, f"asset.{SPLIT}.orig"), "r", encoding="utf-8") as f:
        sources = [line.strip() for line in f.readlines()]
    ref_lists = []
    for i in range(10):
        with open(os.path.join(asset_dir, f"asset.{SPLIT}.simp.{i}"), "r", encoding="utf-8") as f:
            ref_lists.append([line.strip() for line in f.readlines()])
    data = []
    for idx, src in enumerate(sources):
        data.append({
            "complex_raw": src,
            "simple": [ref_lists[r][idx] for r in range(10)],
        })
    return data


def build_eval_dataset(raw, tokenizer):
    """Attach chat-templated prompts using the given model's tokenizer."""
    rows = []
    for item in raw:
        messages = [
            {"role": "system",
             "content": "Simplify the following sentence. Output ONLY the simplified text. Do not explain your answer."},
            {"role": "user", "content": item["complex_raw"]},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        rows.append({"prompt": prompt_text, "complex_raw": item["complex_raw"], "simple": item["simple"]})
    ds = Dataset.from_list(rows)
    return ds.filter(lambda x: 5 < len(x["complex_raw"].split()) < 100)


# ==========================================
# 3. Discover the runs to evaluate (only *-Final adapters)
# ==========================================
def discover_runs(runs_dir, only=None):
    runs = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*-Final"))):
        if not os.path.isdir(path):
            continue
        run = os.path.basename(path)[:-len("-Final")]  # strip trailing -Final
        if only and only not in run:
            continue
        parts = run.split("_")
        if len(parts) < 3:
            print(f"[skip] cannot parse run name '{run}'")
            continue
        label, trainer, reward = parts[0], parts[1], "_".join(parts[2:])
        if label not in BASE_MODELS:
            print(f"[skip] '{run}': unknown model label '{label}' (not in BASE_MODELS)")
            continue
        runs.append({"run": run, "adapter": path, "label": label,
                     "trainer": trainer, "reward": reward})
    return runs


# ==========================================
# 4. Evaluate one adapter
# ==========================================
def evaluate_adapter(entry, raw, metric_bertscore, metric_meaningbert):
    run = entry["run"]
    base_path = BASE_MODELS[entry["label"]]
    print("\n" + "=" * 60)
    print(f"EVAL: {run}   base={base_path}")
    print("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    # Tokenizer saved alongside the adapter during training; fall back to base.
    tok_src = entry["adapter"] if os.path.exists(os.path.join(entry["adapter"], "tokenizer_config.json")) else base_path
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, entry["adapter"])
    model.eval()
    try:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass

    eval_dataset = build_eval_dataset(raw, tokenizer)
    print(f"Evaluating on {len(eval_dataset)} examples...")

    gen_kwargs = {
        "max_new_tokens": 64,
        "min_new_tokens": 5,
        "do_sample": True,
        "temperature": 0.9,
        "top_p": 0.95,
        "repetition_penalty": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ],
    }

    sources, references, predictions = [], [], []
    for i, row in tqdm(enumerate(eval_dataset), total=len(eval_dataset)):
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        output_text = clean_output(output_text)
        sources.append(row["complex_raw"])
        predictions.append(output_text)
        references.append(row["simple"])
        if i < 2:
            print(f"  [{i}] SRC: {row['complex_raw']}\n      PRED: {output_text}")

    # Free the model before metric computation.
    del model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Metrics ----
    formatted_refs_for_sari = list(map(list, zip(*references)))
    try:
        sari_score = corpus_sari(sources, predictions, formatted_refs_for_sari)
    except Exception as e:
        print(f"SARI failed: {e}")
        sari_score = 0.0

    fkgl_scores = [textstat.textstat.flesch_kincaid_grade(p) for p in predictions]
    mean_fkgl = float(np.mean(fkgl_scores))
    mean_length = float(np.mean([len(p.split()) for p in predictions]))

    bs_f1 = None
    try:
        bs_res = metric_bertscore.compute(predictions=predictions, references=references, lang="en")
        bs_f1 = bs_res["f1"]
        bertscore_f1_mean = float(np.mean(bs_f1))
    except Exception as e:
        print(f"BERTScore failed: {e}")
        bertscore_f1_mean = 0.0

    sample_means = None
    try:
        flat_preds, flat_refs = [], []
        for pred, refs in zip(predictions, references):
            for r in refs:
                flat_preds.append(pred)
                flat_refs.append(r)
        mb_res = metric_meaningbert.compute(predictions=flat_preds, references=flat_refs)
        reshaped = np.array(mb_res["scores"]).reshape(len(predictions), 10)
        sample_means = np.max(reshaped, axis=1)
        meaningbert_mean = float(np.mean(sample_means))
    except Exception as e:
        print(f"MeaningBERT failed: {e}")
        meaningbert_mean = 0.0

    # ---- Per-run outputs (named by run) ----
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{run}.txt"), "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(str(p).strip() + "\n")
    if bs_f1 is not None:
        pd.DataFrame(bs_f1).to_csv(os.path.join(OUT_DIR, f"{run}-bertscores.csv"), index=False, header=False)
    if sample_means is not None:
        pd.DataFrame(sample_means).to_csv(os.path.join(OUT_DIR, f"{run}-meaningbert.csv"), index=False, header=False)

    print(f"  SARI={sari_score:.2f}  FKGL={mean_fkgl:.2f}  Len={mean_length:.1f}  "
          f"BERTScore={bertscore_f1_mean:.4f}  MeaningBERT={meaningbert_mean:.4f}")

    return {
        "run": run, "model": entry["label"], "trainer": entry["trainer"], "reward": entry["reward"],
        "SARI": round(sari_score, 2), "FKGL": round(mean_fkgl, 2), "Length": round(mean_length, 1),
        "BERTScore_F1": round(bertscore_f1_mean, 4), "MeaningBERT": round(meaningbert_mean, 4),
        "n": len(predictions),
    }


# ==========================================
# 5. Main: iterate all *-Final adapters
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs", help="Folder containing *-Final adapters")
    parser.add_argument("--out-dir", default="System_Output", help="Folder for system outputs")
    parser.add_argument("--only", default=None, help="Substring filter: eval only matching run names")
    args = parser.parse_args()

    global RUNS_DIR, OUT_DIR
    RUNS_DIR, OUT_DIR = args.runs_dir, args.out_dir

    runs = discover_runs(RUNS_DIR, only=args.only)
    if not runs:
        print(f"No *-Final adapters found in '{RUNS_DIR}'.")
        return
    print(f"Found {len(runs)} adapters to evaluate:")
    for r in runs:
        print(f"  - {r['run']}")

    raw = load_asset_raw()
    print(f"Loaded {len(raw)} ASSET ({SPLIT}) examples.")

    print("Loading metric models (once)...")
    metric_bertscore = evaluate.load("bertscore")
    metric_meaningbert = evaluate.load("davebulaval/meaningbert")

    summary_rows = []
    for entry in runs:
        try:
            summary_rows.append(
                evaluate_adapter(entry, raw, metric_bertscore, metric_meaningbert))
        except Exception as e:
            print(f"[FAIL] {entry['run']}: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if summary_rows:
        os.makedirs(OUT_DIR, exist_ok=True)
        df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(OUT_DIR, "metrics_summary.csv")
        df.to_csv(summary_path, index=False)
        print("\n" + "=" * 60)
        print("SUMMARY (all runs)")
        print("=" * 60)
        print(df.to_string(index=False))
        print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
