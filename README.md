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

## Setup

1. Clone this repo.
2. Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate promise-java-extract
```
3. Download PROMISE dataset and place under `data/raw/` following the structure in `config.yaml`.

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


