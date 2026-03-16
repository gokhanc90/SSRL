import difflib

import evaluate
import numpy
import torch

from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig
from datasets import  Dataset
from trl import  GRPOConfig, GRPOTrainer
import textstat

from easse.sari import corpus_sari
from trl.models import prepare_model_for_kbit_training

import re


def clean_output(text):
    # 1. Remove common preambles using Regex
    # Matches "Sure, here is...", "The simplified version is:", etc.
    patterns = [
        r"^Sure, here is (the )?simplified sentence:?\s*",
        r"^Here is the simplification:?\s*",
        r"^The simplified text is:?\s*",
        r"^Simplify:?\s*",
    ]

    cleaned = text
    for p in patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)

    # 2. Remove quotes if the model wrapped the output in them
    cleaned = cleaned.strip().strip('"').strip("'")

    # 3. Cut off if it starts rambling after a newline (common in Llama-3)
    if "\n" in cleaned:
        cleaned = cleaned.split("\n")[0]

    return cleaned.strip()
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
        refs_for_this_sentence = [ref_lists[r][idx] for r in range(10)]

        # 1. Define the conversation
        messages = [
            # System Prompt: STRICTLY constraints the behavior
            {"role": "system",
             "content": "Simplify the following sentence. Output ONLY the simplified text. Do not explain your answer."},
            {"role": "user", "content": src}
        ]

        # 2. Apply template (converts list to Llama-3 string format)
        # We use tokenizer.apply_chat_template WITHOUT generating tensors yet
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        data_list.append({
            "prompt": prompt_text,  # Now contains <|begin_of_text|><|start_header_id|>...
            "complex_raw": src,
            "simple": refs_for_this_sentence
        })

    return Dataset.from_list(data_list)
# ==========================================
# 3. Define Reward Functions (The Magic Part)
# ==========================================
# GRPO automatically passes columns from your dataset to these functions.
# Since your dataset has 'simple', we can accept it as an argument!
def meaningBert_reward_func(completions, complex_raw, simple, **kwargs):

    rewards = []

    # 1. Batch inputs for faster MeaningBERT inference
    # MeaningBERT is slow; processing in batch is 10x faster than loops
    all_preds = []
    all_refs = []

    # We need to track which ref belongs to which completion for scoring later
    # But MeaningBERT takes [pred, ref] pairs.
    # Since we have 10 refs, we compare against the BEST matching ref (max score)
    # OR we can compare against the Input for a quick sanity check, but let's stick to refs.

    # Pre-clean and prepare batch
    clean_preds = [clean_output(p) for p in completions]

    # 2. Check for Copying (The Critical Fix)
    # We calculate a simple "Copy Penalty" mask
    copy_penalties = []
    for pred, src in zip(clean_preds, complex_raw):
        similarity = difflib.SequenceMatcher(None, pred, src).ratio()

        # If prediction is identical to source (ignoring case/whitespace)
        if pred.lower().strip() == src.lower().strip():
            copy_penalties.append(-1.0)  # Massive penalty for copying
        elif len(pred) == 0:
            copy_penalties.append(-1.0)  # Penalty for empty
        elif similarity > 0.95:
            copy_penalties.append(-1.0)
        else:
            copy_penalties.append(0.0)

    # 3. Prepare MeaningBERT inputs
    # We flatten: pred1->ref1, pred1->ref2... to find the best semantic match
    flat_preds = []
    flat_refs = []

    # We will score every prediction against ALL its 10 references to find the max
    for pred, refs in zip(clean_preds, simple):
        for r in refs:
            flat_preds.append(pred)
            flat_refs.append(r)

    # 4. Compute MeaningBERT Scores (Batch)
    try:
        results = meaning_bert.compute(predictions=flat_preds, references=flat_refs)
        all_scores = numpy.array(results['scores'])  # Shape: (batch * 10,)

        # Reshape to (batch, 10) to find max score per prediction
        scores_matrix = all_scores.reshape(len(completions), 10)
        max_scores = numpy.max(scores_matrix, axis=1)  # Shape: (batch,)

    except Exception as e:
        print(f"MeaningBERT Failed: {e}")
        max_scores = numpy.zeros(len(completions))

    # 5. Combine Scores
    for i in range(len(completions)):
        bert_score = max_scores[i] / 100.0  # Normalize 0-100 to 0-1
        penalty = copy_penalties[i]

        # Add Length Bonus (Simplification usually implies shortening)
        # Reward = Meaning_Score * (1 + Length_Reduction_Bonus)
        src_len = len(complex_raw[i].split())
        pred_len = len(clean_preds[i].split())

        # Bonus if length is reduced by 10-40%
        len_ratio = pred_len / max(1, src_len)
        if 0.6 <= len_ratio <= 0.9:
            len_bonus = 0.1
        elif len_ratio > 1.0:
            len_bonus = -0.1  # Penalty for getting longer
        else:
            len_bonus = 0.0

        final_reward = bert_score + penalty + len_bonus
        rewards.append(final_reward)

    return rewards




def sari_reward_func(completions, complex_raw, simple, **kwargs):
    """
    Calculates SARI using ASSET's 10 references.
    Args:
        simple: List[List[str]] -> Each item is a list of 10 refs.
    """
    rewards = []
    for src, pred, refs in zip(complex_raw, completions, simple):
        # EASSE expects: references = [[ref1], [ref2], ..., [ref10]] (Shape: 10 x 1)
        # for a single sentence comparison.
        # We transform our list ['r1', 'r2'...] into [['r1'], ['r2']...]
        formatted_refs = [[r] for r in refs]

        try:
            # Calculate SARI (0-100)
            score = corpus_sari([src], [pred], formatted_refs)
        except Exception:
            score = 0.0

        # Normalize 0-100 -> 0.0-1.0
        rewards.append(score / 100.0)

    return rewards



def run(outputfolder,rewards):
    global SPLIT, meaning_bert, tokenizer
    # ... [Load Model, Tokenizer, and Dataset as defined in previous steps] ...
    # ==========================================
    # 1. Configuration & Model Loading
    # ==========================================

    MODEL_NAME = "./Llama-3.2-3B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # LoRA Config (Passed to Trainer)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",  # Optional: Faster attention for Llama-3
        torch_dtype=torch.float32
    )
    model = prepare_model_for_kbit_training(model)

    model.generation_config.do_sample = True
    model.generation_config.temperature = 0.9
    model.generation_config.top_p = 0.95
    model.generation_config.repetition_penalty = 1.0  
    model.generation_config.max_new_tokens = 64
    model.generation_config.pad_token_id = tokenizer.pad_token_id

   
    model.generation_config.eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Load Dataset
    SPLIT = "valid"  # Use 'valid' (2000 samples) for GRPO tuning. 'test' is for final eval.
    dataset = create_asset_dataset()
    print(f"Loaded {len(dataset)} examples from ASSET ({SPLIT}).")
    
    print(f"Loaded dataset with {len(dataset)} examples.")
    print("First 5 examples:")
    for i in range(min(5, len(dataset))):
        print(f"Complex: {dataset[i]['prompt']}")
        print(f"Simple: {dataset[i]['simple']}")
        print(f"Simple: {dataset[i]['complex_raw']}")

    # Filter length outliers to stabilize training
    dataset = dataset.filter(lambda x: 5 < len(x['complex_raw'].split()) < 100)

    # ==========================================

    meaning_bert = evaluate.load("davebulaval/meaningbert")


    # ==========================================
    # 4. GRPO Trainer Initialization
    # ==========================================
    training_args = GRPOConfig(
        output_dir=outputfolder,
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_generations=4,  
        max_completion_length=64,  
        bf16=False,  
        logging_steps=1000,

        num_train_epochs=3,  
        save_strategy="epoch",  

    )

    trainer = GRPOTrainer(
        model=model,  
        reward_funcs=rewards,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer, 
    )

    # ==========================================
    # 5. Train
    # ==========================================
    print("Starting GRPO Training...")
    trainer.train()
    print("Training Complete.")
    print("Saving final model...")
    FINAL_DIR = f"{outputfolder}-Final"

    # Save Adapter
    trainer.save_model(FINAL_DIR)
    # Save Tokenizer
    tokenizer.save_pretrained(FINAL_DIR)

    print(f"Training Complete. Model and Tokenizer are saved to {FINAL_DIR}")


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--Output", help="Output folder for saving Model")
parser.add_argument("-r", "--Reward", help="List of Reward functions with comma separated: sari_reward_func,meaningBert_reward_func")

args = parser.parse_args()
print(args)
rewards = [eval(a) for a in args.Reward.split(",")]
if args.Output:
    print("Output:", args.Output)
run(args.Output, rewards)