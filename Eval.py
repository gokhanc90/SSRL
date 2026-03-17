import pandas as pd

from easse.cli import get_orig_and_refs_sents, get_sys_sents
from easse.sari import corpus_sari
from easse.fkgl import corpus_fkgl
import evaluate
import numpy as np


def run(DATASET_NAME,METHOD):
    file_path = f'easse/easse/resources/data/system_outputs/{DATASET_NAME}/test/{METHOD}'

    predictions = get_sys_sents(f'{DATASET_NAME}_test', sys_sents_path=file_path)
    print(f"Loaded {len(predictions)} predictions.")

    # ==========================================
    # 2. Load the ASSET Test Set (Sources & Refs)
    # ==========================================
    print(f"Loading {DATASET_NAME} test set references...")

    # In easse, the task name for the ASSET test set is "asset_test"
    sources, references = get_orig_and_refs_sents(f"{DATASET_NAME}_test",
                                                  orig_sents_path=None,
                                                  refs_sents_paths=None
                                                  )
    print(f"Loaded {len(sources)} sentences.")
    print(f"Found {len(references)} reference sets (annotators).")
    # ==========================================
    # 3. Calculate Metrics
    # ==========================================
    print("\nCalculating metrics...\n")


    sari_score = corpus_sari(sources, predictions, references)
    fkgl_score = corpus_fkgl(predictions)


    lengths = [len(p.split()) for p in predictions]
    mean_length = np.mean(lengths)

    row_major_refs = list(map(list, zip(*references)))

    try:
        print("Computing BERTScore...")
        bertscore = evaluate.load("bertscore")
        bs_res = bertscore.compute(predictions=predictions, references=row_major_refs, lang="en")
        bertscore_f1_mean = np.mean(bs_res["f1"])
    except Exception as e:
        print(f"BERTScore failed: {e}")
        bertscore_f1_mean = 0.0

    try:
        print("Computing MeaningBERT (this may take time)...")
        meaning_bert = evaluate.load("davebulaval/meaningbert")

        # Flatten -> Compute -> Reshape -> Mean
        flat_preds = []
        flat_refs = []
        num_refs_per_sample = len(row_major_refs[0])  # Should be 8 for TurkCorpus

        for pred, refs in zip(predictions, row_major_refs):
            for r in refs:
                flat_preds.append(pred)
                flat_refs.append(r)

        mb_res = meaning_bert.compute(predictions=flat_preds, references=flat_refs)

        raw_scores = np.array(mb_res['scores'])
        # Reshape based on actual number of references found
        reshaped_scores = raw_scores.reshape(len(predictions), num_refs_per_sample)
        sample_means = np.max(reshaped_scores, axis=1)
        meaningbert_mean = np.mean(sample_means)

    except Exception as e:
        print(f"MeaningBERT failed: {e}")
        meaningbert_mean = 0.0

    sacrebleu = evaluate.load("sacrebleu")

    # SacreBLEU requires Row-Major references: [ [ref1_s1, ref2_s1], [ref1_s2, ref2_s2] ]
    bleu_res = sacrebleu.compute(predictions=predictions, references=row_major_refs)
    bleu_score = bleu_res["score"]  # Scale is 0 to 100


    sources_as_refs = [[s] for s in sources]
    bleu_oi_res = sacrebleu.compute(predictions=predictions, references=sources_as_refs)
    bleu_oi_score = bleu_oi_res["score"]

    # 8. iBLEU (Interpolated BLEU)
    alpha = 0.9
    ibleu_score = (alpha * bleu_score) - ((1 - alpha) * bleu_oi_score)

    # 9. FKBLEU
    fkbleu_score = ibleu_score - fkgl_score

    print("\n" + "=" * 30)
    print(f"FINAL EVALUATION RESULTS ({DATASET_NAME} {METHOD} )")
    print("=" * 30)
    print(f"SARI Score        :\t{sari_score:.2f}")
    print(f"FKGL (Grade)      :\t{fkgl_score:.2f}")
    print(f"Avg Length (words) :\t{mean_length:.1f}")
    print(f"BLEU Score        :\t{bleu_score:.2f}")
    print(f"iBLEU Score       :\t{ibleu_score:.2f}")
    print(f"FKBLEU Score      :\t{fkbleu_score:.2f}")
    print(f"BERTScore F1      :\t{bertscore_f1_mean:.4f}")
    print(f"MeaningBERT Score :\t{meaningbert_mean:.4f}")

    print("=" * 30)

    # Save requested files
    print("Saving output files...")

    with open(f"results/{DATASET_NAME}-{METHOD}-complex.txt", "w", encoding="utf-8") as f:
        for s in sources:
            f.write(str(s).strip() + "\n")

    with open(f"results/{DATASET_NAME}-{METHOD}-predictions.txt", "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(str(p).strip() + "\n")

    if 'bs_res' in locals() and 'f1' in bs_res:
        pd.DataFrame(bs_res["f1"]).to_csv(f"results/{DATASET_NAME}-{METHOD}-bertscores.csv", index=False, header=False)

    if 'sample_means' in locals():
        pd.DataFrame(sample_means).to_csv(f"results/{DATASET_NAME}-{METHOD}-meaningbert.csv", index=False, header=False)



ds=["turkcorpus","asset"]  # asset
mts = ["predictions","Dress","Dress-Ls","Hybrid","ACCESS"]

for d in ds:
    for m in mts:
        run(d,m)
