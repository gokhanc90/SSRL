import os
import glob

import numpy as np
import pandas as pd
import evaluate
import easse

from easse.cli import get_orig_and_refs_sents, get_sys_sents
from easse.sari import corpus_sari
from easse.fkgl import corpus_fkgl

# ==========================================
# Config
# ==========================================
SYSOUT_DIR = "System_Output"   # trained-model outputs: <run>.txt
RESULTS_DIR = "results"
DATASETS = ["asset", "turkcorpus"]
BASELINE_METHODS = ["Dress", "Dress-Ls", "Hybrid", "ACCESS"]
SKIP_NEURAL = os.environ.get("EVAL_SKIP_NEURAL", "0") == "1"  # skip BERTScore/MeaningBERT (fast check)

# ASSET & TurkCorpus test sources are identical -> one prediction file per method,
# scored against each dataset's own references. Baselines ship under turkcorpus/test.
_EASSE_SO = os.path.join(os.path.dirname(easse.__file__), "resources", "data", "system_outputs")


def discover_model_methods():
    """Trained-model system outputs live in SYSOUT_DIR as <run>.txt."""
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(SYSOUT_DIR, "*.txt")))


def predictions_path(method):
    """One predictions file per method (sources are identical across datasets)."""
    local = os.path.join(SYSOUT_DIR, f"{method}.txt")
    if os.path.exists(local):
        return local                                              # trained model
    return os.path.join(_EASSE_SO, "turkcorpus", "test", method)  # classic baseline same for asset and turkcorpus


def compute_row(method, predictions, sources, references, refs_row_major, metrics):
    sari_score = corpus_sari(sources, predictions, references)
    fkgl_score = corpus_fkgl(predictions)
    mean_length = float(np.mean([len(p.split()) for p in predictions]))

    # --- BERTScore ---
    bs_f1 = None
    bertscore_f1_mean = np.nan
    if not SKIP_NEURAL:
        try:
            bs_res = metrics["bertscore"].compute(predictions=predictions, references=refs_row_major, lang="en")
            bs_f1 = bs_res["f1"]
            bertscore_f1_mean = float(np.mean(bs_f1))
        except Exception as e:
            print(f"  BERTScore failed ({method}): {e}")

    # --- MeaningBERT (max over refs, matching training) ---
    sample_means = None
    meaningbert_mean = np.nan
    if not SKIP_NEURAL:
        try:
            n_refs = len(refs_row_major[0])
            flat_preds, flat_refs = [], []
            for pred, refs in zip(predictions, refs_row_major):
                for r in refs:
                    flat_preds.append(pred)
                    flat_refs.append(r)
            mb_res = metrics["meaningbert"].compute(predictions=flat_preds, references=flat_refs)
            reshaped = np.array(mb_res["scores"]).reshape(len(predictions), n_refs)
            sample_means = np.max(reshaped, axis=1)
            meaningbert_mean = float(np.mean(sample_means))
        except Exception as e:
            print(f"  MeaningBERT failed ({method}): {e}")

    # --- BLEU / iBLEU / FKBLEU ---
    bleu_score = metrics["sacrebleu"].compute(predictions=predictions, references=refs_row_major)["score"]
    bleu_oi = metrics["sacrebleu"].compute(predictions=predictions, references=[[s] for s in sources])["score"]
    alpha = 0.9
    ibleu_score = (alpha * bleu_score) - ((1 - alpha) * bleu_oi)
    fkbleu_score = ibleu_score - fkgl_score

    row = {
        "Method": method,
        "SARI": round(sari_score, 2),
        "FKGL": round(fkgl_score, 2),
        "Length": round(mean_length, 1),
        "BLEU": round(bleu_score, 2),
        "iBLEU": round(ibleu_score, 2),
        "FKBLEU": round(fkbleu_score, 2),
        "BERTScore_F1": (round(bertscore_f1_mean, 4) if not np.isnan(bertscore_f1_mean) else np.nan),
        "MeaningBERT": (round(meaningbert_mean, 4) if not np.isnan(meaningbert_mean) else np.nan),
    }
    return row, bs_f1, sample_means


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    methods = discover_model_methods() + BASELINE_METHODS
    print(f"Evaluating {len(methods)} methods: {methods}")

    if SKIP_NEURAL:
        print("[EVAL_SKIP_NEURAL=1] Skipping BERTScore/MeaningBERT.")
        metrics = {"sacrebleu": evaluate.load("sacrebleu")}
    else:
        print("Loading metric models (once)...")
        metrics = {
            "bertscore": evaluate.load("bertscore"),
            "meaningbert": evaluate.load("davebulaval/meaningbert"),
            "sacrebleu": evaluate.load("sacrebleu"),
        }

    for dataset in DATASETS:
        print(f"\n########## {dataset.upper()} ##########")
        sources, references = get_orig_and_refs_sents(f"{dataset}_test",
                                                      orig_sents_path=None, refs_sents_paths=None)
        refs_row_major = list(map(list, zip(*references)))

        rows = []
        for method in methods:
            path = predictions_path(method)
            if not os.path.exists(path):
                print(f"  [skip] {method}: predictions not found ({path})")
                continue
            predictions = get_sys_sents(f"{dataset}_test", sys_sents_path=path)
            if len(predictions) != len(sources):
                print(f"  [warn] {method}: {len(predictions)} preds != {len(sources)} sources")

            row, bs_f1, sample_means = compute_row(
                method, predictions, sources, references, refs_row_major, metrics)
            rows.append(row)

            # per-method detail CSVs (consumed by TRisk analysis)
            if bs_f1 is not None:
                pd.DataFrame(bs_f1).to_csv(
                    f"{RESULTS_DIR}/{dataset}-{method}-bertscores.csv", index=False, header=False)
            if sample_means is not None:
                pd.DataFrame(sample_means).to_csv(
                    f"{RESULTS_DIR}/{dataset}-{method}-meaningbert.csv", index=False, header=False)

            print(f"  {method:24s} SARI={row['SARI']:.2f}  FKGL={row['FKGL']:.2f}  BLEU={row['BLEU']:.2f}")

        df = pd.DataFrame(rows)
        out = os.path.join(RESULTS_DIR, f"{dataset}_metrics.csv")
        df.to_csv(out, index=False)
        print("\n" + "=" * 40)
        print(f"{dataset.upper()} METRICS TABLE")
        print("=" * 40)
        print(df.to_string(index=False))
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
