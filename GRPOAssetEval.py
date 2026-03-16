import torch
import numpy as np
import textstat
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel
from easse.sari import corpus_sari  # Ensure this matches your import path
import re
import evaluate
import pandas as pd

# ==========================================
# 1. Configuration
# ==========================================
MODEL_NAME = "Llama-3.2-3B-Instruct"


ADAPTER_PATH = "GRPOModelt9tp95rp1-Final"
def clean_output(text):
    # 1. Remove common preambles using Regex
    patterns = [
        r"^Sure, here is (the )?simplified sentence:?\s*",
        r"^Here is the simplification:?\s*",
        r"^The simplified text is:?\s*",
        r"^Simplify:?\s*",
    ]

    cleaned = text
    for p in patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)

    # Remove surrounding quotes
    cleaned = cleaned.strip().strip('"').strip("'")

    # If model rambles into newlines, keep only first line
    if "\n" in cleaned:
        cleaned = cleaned.split("\n")[0]

    return cleaned.strip()


# ==========================================
# 2. Load Model & Tokenizer
# ==========================================
print("Loading Base Model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


# 1. Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
)

# 2. Load Trained Adapter (LoRA)
print(f"Loading Adapter from {ADAPTER_PATH}...")
try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
except Exception as e:
    print(f"Error loading adapter: {e}")
    print("Ensure ADAPTER_PATH points to the folder containing 'adapter_model.bin' or 'adapter_config.json'")
    exit()

model.eval()  # Set to evaluation mode

try:
    model.generation_config.pad_token_id = tokenizer.pad_token_id
except Exception:
    # Some PeftModel wrappers may not expose generation_config; gen_kwargs will be used below
    pass


# ==========================================
# 3. Load Dataset (Same logic as Training)
# ==========================================
SPLIT = "test" # Use 'valid' (2000 samples) for GRPO tuning. 'test' is for final eval.
def create_asset_dataset():
    # 1. Download Files if missing
    files = {
        "orig": f"asset/dataset/asset.{SPLIT}.orig",
    }
    # Add the 10 reference files
    for i in range(10):
        files[f"simp.{i}"] = f"asset/dataset/asset.{SPLIT}.simp.{i}"

    # 2. Load Content
    print("Loading ASSET dataset...")
    with open(files["orig"], "r", encoding="utf-8") as f:
        sources = [line.strip() for line in f.readlines()]

    # Load all 10 references into a list of lists: [[ref0_line0, ...], [ref0_line1, ...]]
    ref_lists = []
    for i in range(10):
        with open(files[f"simp.{i}"], "r", encoding="utf-8") as f:
            ref_lists.append([line.strip() for line in f.readlines()])

    # 3. Structure Data
    # We need to pivot so each sample has [ref0, ref1, ..., ref9]
    data_list = []
    for idx, src in enumerate(sources):
        # Gather all 10 refs for this specific sentence
        refs_for_this_sentence = [ref_lists[r][idx] for r in range(10)]
        
        messages = [
            {"role": "system",
             "content": "Simplify the following sentence. Output ONLY the simplified text. Do not explain your answer."},
            {"role": "user", "content": src}
        ]
        
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        
        data_list.append({
            "prompt": prompt_text,
            "complex_raw": src,
            "simple": refs_for_this_sentence
        })
#    data_list = data_list[0:10]
    return Dataset.from_list(data_list)


# Load Dataset
dataset = create_asset_dataset()
print(f"Loaded {len(dataset)} examples from ASSET ({SPLIT}).")

# Filter length outliers to stabilize training
eval_dataset = dataset.filter(lambda x: 5 < len(x['complex_raw'].split()) < 100)


print(f"Evaluating on {len(eval_dataset)} examples...")

# ==========================================
# 4. Generation & Evaluation Loop
# ==========================================
sources = []
references = []
predictions = []

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
        tokenizer.convert_tokens_to_ids("<|eot_id|>")  
    ]
}

print("Starting Generation...")
for i, row in tqdm(enumerate(eval_dataset), total=len(eval_dataset)):
    prompt = row["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Slice off the prompt to get only the generated simplification
    output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    output_text = clean_output(output_text)

    # Collect data for metrics
    sources.append(row["complex_raw"])
    predictions.append(output_text)
    references.append(row["simple"])  # SARI expects List[List[str]] for references

    # Print first 3 examples to sanity check
    if i < 3:
        print(f"\n--- Example {i + 1} ---")
        print(f"Source: {row['complex_raw']}")
        print(f"Ref   : {row['simple']}")
        print(f"Pred  : {output_text}")


del model
# ==========================================
# 5. Outputs
# ==========================================
print("\n" + "=" * 30)
print("COMPLEX-RAW")
for t in sources:
    print(t)

print("\n" + "=" * 30)

print("PREDICTIONS")
for t in predictions:
    print(t)

print("\n" + "=" * 30)
# ==========================================
# 5. Calculate Metrics
# ==========================================
print("\nCalculating Metrics...")
formatted_refs_for_sari = list(map(list, zip(*references)))
# 1. SARI
try:
    sari_score = corpus_sari(sources, predictions, formatted_refs_for_sari)
except Exception as e:
    print(f"SARI Calculation Failed: {e}")
    sari_score = 0.0

# 2. FKGL (Readability)
fkgl_scores = [textstat.textstat.flesch_kincaid_grade(p) for p in predictions]
mean_fkgl = np.mean(fkgl_scores)

# 3. Length
lengths = [len(p.split()) for p in predictions]
mean_length = np.mean(lengths)

# --- 3. BERTScore ---
# We compare Prediction vs References (Standard Quality Eval)
try:
    print("Computing BERTScore...")
    bertscore = evaluate.load("bertscore")
    # BERTScore handles multi-ref automatically by taking max similarity usually
    bs_res = bertscore.compute(predictions=predictions, references=references, lang="en")
    bertscore_f1_mean = np.mean(bs_res["f1"])
except Exception as e:
    print(f"BERTScore failed: {e}")
    bertscore_f1_mean = 0.0

# --- 4. MeaningBERT ---
# Matching Training Logic: Average score against all 10 references
try:
    print("Computing MeaningBERT (this may take time)...")
    meaning_bert = evaluate.load("davebulaval/meaningbert")
    
    # We must flatten to batch process, or loop. Looping is safer for exact logic match.
    # Training logic: avg(score(pred, ref1), score(pred, ref2)...)
    
    all_mb_scores = []
    
    # Batch processing for speed (Flatten -> Compute -> Reshape -> Mean)
    flat_preds = []
    flat_refs = []
    for pred, refs in zip(predictions, references):
        for r in refs:
            flat_preds.append(pred)
            flat_refs.append(r)
            
    # Compute in one go
    mb_res = meaning_bert.compute(predictions=flat_preds, references=flat_refs)
    
    # Reshape and average per sample
    # mb_res['scores'] is a list of N * 10 scores
    raw_scores = np.array(mb_res['scores'])
    # Reshape to (N_samples, 10_refs)
    reshaped_scores = raw_scores.reshape(len(predictions), 10)
    # Average across the 10 refs for each sample
    sample_means = np.max(reshaped_scores, axis=1)
    
    meaningbert_mean = np.mean(sample_means)

except Exception as e:
    print(f"MeaningBERT failed: {e}")
    meaningbert_mean = 0.0


print("\n" + "=" * 30)
print(f"FINAL EVALUATION RESULTS")
print("=" * 30)
print(f"SARI Score  : {sari_score:.2f}")
print(f"FKGL (Grade): {mean_fkgl:.2f}")
print(f"Avg Length  : {mean_length:.1f} words")
print(f"BERTScore F1 (avg): {bertscore_f1_mean:.4f}")
print(f"MeaningBERT (avg): {meaningbert_mean:.4f}") 
print("=" * 30)

    # Save requested files
print("Saving output files...")
    
with open("proposed-predictions.txt", "w", encoding="utf-8") as f:
    for p in predictions:
        f.write(str(p).strip() + "\n")


if 'bs_res' in locals() and 'f1' in bs_res:
    pd.DataFrame(bs_res["f1"]).to_csv("asset-proposed-bertscores.csv", index=False, header=False)

if 'sample_means' in locals():
    pd.DataFrame(sample_means).to_csv("asset-proposed-meaningbert.csv", index=False, header=False)