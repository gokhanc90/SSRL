# A Comprehensive Study of Sentence Simplification with Critic-Free Reinforcement Learning

Reproduction code for a multi-objective, **critic-free** reinforcement-learning approach to
sentence simplification. The framework fine-tunes small Llama models with a combined reward
(SARI + MeaningBERT with an output-length brevity term) and studies the problem along **three
design axes**:

| Axis | Values |
|------|--------|
| Model size | Llama-3.2 **1B** / **3B** (Instruct) |
| RL trainer (critic-free) | **GRPO** / **RLOO** |
| Reward | **SARI** / **MeaningBERT (MB)** / **SARI+MB** (combined = full system) |

Efficiency: 4-bit NF4 quantization + LoRA, so the full 2×2×3 grid runs on a single consumer GPU.
Both trainers share the same learning rate, batch size, and number of sampled generations, and gains
are validated with a risk-sensitive **TRisk** analysis (whose risk-neutral level equals a
paired-samples *t*-test). The KL coefficient is left at the `trl` defaults (see Installation).

## Repository structure

```
.
├── train_asset.py              # RL training: {GRPO|RLOO} x {1B|3B} x {reward combo}, LoRA + 4-bit
├── run_grid.sh                 # runs the full 2x2x3 grid sequentially (env-overridable)
├── generate_outputs.py         # inference: trained adapters -> System_Output/<run>.txt (+ quick metrics)
├── Eval.py                     # SARI/BLEU/iBLEU/FKBLEU/BERTScore/MeaningBERT tables + per-sentence CSVs
├── TRiskEvaluation.py          # TRisk statistic (risk-sensitive), alpha=0 == paired t-test
├── TRiskEvaluationAnalysis.py  # TRisk tables (alpha 0..5) for the combined-reward models vs baselines
├── TRiskPlotGenerator.py       # TRisk 2x2 curve plots per dataset
├── requirements.txt
└── System_Output/              # full-system (SARI+MB combined) outputs for the 4 backbones
    ├── llama1b_grpo_sari-mb.txt
    ├── llama1b_rloo_sari-mb.txt
    ├── llama3b_grpo_sari-mb.txt
    └── llama3b_rloo_sari-mb.txt
```

> **Note on `System_Output/`.** Only the four **full-system** (combined `SARI+MB`) outputs are
> shipped here. The reward-ablation outputs (`*_sari.txt`, `*_mb.txt`) are not included; regenerate
> them with `run_grid.sh` + `generate_outputs.py` if you want to reproduce the ablation tables.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**TRL version and KL coefficient.** The reported runs used `trl==0.25.1` (pinned in
`requirements.txt`). `train_asset.py` does not set the KL coefficient `beta`, so the `trl` defaults
apply: `0.0` for GRPO (no KL term, no reference model) and `0.05` for RLOO (KL penalty subtracted
from the reward).

**EASSE** (SARI/FKGL metrics + the ASSET & TurkCorpus test sets and classical baseline outputs) is
installed from source:

```bash
git clone https://github.com/feralvam/easse.git
pip install -e easse
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

**Base models.** `train_asset.py` (`MODELS`) and `generate_outputs.py` (`BASE_MODELS`) point to the
gated Hugging Face repos `meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-3.2-3B-Instruct`
(run `huggingface-cli login`, or edit the dicts to local paths).

## Reproduction

### 1. Train the grid (2 models × 2 trainers × 3 rewards)

```bash
bash run_grid.sh
```

Each run writes a LoRA adapter to `runs/<model>_<trainer>_<reward>-Final/`. Override any axis:

```bash
MODELS="llama-1b" TRAINERS="grpo" bash run_grid.sh   # subset
DRY_RUN=1 bash run_grid.sh                            # list runs only
```

A single run directly:

```bash
python train_asset.py -m llama-3b -t grpo \
    -r sari_reward_func,meaningBert_reward_func \
    -o runs/llama3b_grpo_sari-mb
```

Flags: `-m {llama-1b,llama-3b}`, `-t {grpo,rloo}`, `-r` comma-separated reward funcs
(`sari_reward_func`, `meaningBert_reward_func`), `-d {cuda,cpu}`, `-s` (smoke test).

### 2. Generate system outputs

```bash
python generate_outputs.py            # discovers runs/*-Final, writes System_Output/<run>.txt
```

### 3. Evaluate (metric tables)

```bash
python Eval.py                        # -> results/asset_metrics.csv, results/turkcorpus_metrics.csv
EVAL_SKIP_NEURAL=1 python Eval.py     # fast: SARI/BLEU/FKGL only
```

`Eval.py` scores every method in `System_Output/` plus the classical baselines (Hybrid, DRESS,
DRESS-Ls, ACCESS) bundled with EASSE, and also writes per-sentence `results/<dataset>-<method>-{bertscores,meaningbert}.csv` used by the TRisk step.

### 4. Risk-sensitive (TRisk) analysis

```bash
python TRiskEvaluationAnalysis.py     # -> results/TRisk_Output/*.csv (alpha 0..5 tables + p-values)
python TRiskPlotGenerator.py          # -> figures/TRisk_Plot_{asset,turkcorpus}.eps (2x2 grids)
```

ASSET and TurkCorpus share identical source sentences, so one prediction file per system is scored
against each dataset's own references.
