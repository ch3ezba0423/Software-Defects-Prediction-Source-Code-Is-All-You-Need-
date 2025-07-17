# Software Defects Prediction: Code Is All You Need

This repository reproduces the experiments from our paper:
"Software Defects Prediction: Source Code Is All You Need"

## Authors

- **Thierno Aliou Ba** — thierno.aliou.ba@uqtr.ca  
  Master’s student, UQTR
- **Fadel Toure** — fadel.toure@uqtr.ca  
  Research supervisor, UQTR
- **Usef Faghihi** — usef.faghihi@uqtr.ca  
  Research supervisor, UQTR


## Repository Structure

```
./                              # root of repo
├── README.md                   # project overview and setup instructions
├── environment.yml             # conda environment spec
├── config.yaml                 # paths for source & metrics
├── .gitignore                  # ignore Python artifacts, env folders, outputs
├── scripts/                    # Python scripts for each pipeline step
│   ├── extract_classes.py      # extraction from JAVA → DataFrame (uses config.yaml)
│   ├── preprocess.py           # remove comments, replace strings, format code
│   ├── dedup_and_sample.py     # deduplication & selective undersampling
│   ├── tokenize.py             # adding contextual token if necessary and tokenization (DeepSeek Coder tokenizer)
│   ├── train_and_val.py        # LOPOCV training & evaluation loops
├── notebooks/                  # exploratory analysis & final result reports
│   └── LOPOCV_Results.ipynb    # notebook for loading and visualizing LOPOCV results
└── outputs/                    # generated outputs
    ├── logs/                   # training logs, tokenizer logs
    └── results/                # confusion matrices, tables, charts
```

## Setup

1. Clone this repo.
2. Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate promise-java-extract
```
3. Download [PROMISE dataset](https://github.com/feiwww/PROMISE-backup) and place under `data/raw/` following the structure in `config.yaml`.

## Pipeline

Each step can be run sequentially from the root:

```bash
python scripts/extract.py --config config.yaml
python scripts/preprocess.py --config config.yaml
python scripts/dedup_and_sample.py --config config.yaml
python scripts/tokenize.py --config config.yaml
python scripts/train_and_eval.py --config config.yaml
```

Results and logs will be saved in `outputs/`.


