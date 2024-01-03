# PIRL-ARI5004-Deep-Learning

# Self-Supervised Learning with PIRL: Reproduction Instructions

This repository contains the implementation of the paper "Self-Supervised Learning of Pretext-Invariant Representations" by Ishan Misra and Laurens van der Maaten. The goal is to reproduce the experiment results presented in the ARI5004-Deep-Learning lecture project report.

## Prerequisites

- Python 3.x
- PyTorch (installation instructions [here](https://pytorch.org/get-started/locally/))
- Other dependencies (install via `pip install -r requirements.txt`)

## Project Structure

The project is structured as follows:

- `src/`: Source code for the implementation.
- `data/`: Placeholder for dataset (not included).
- `results/`: Placeholder for storing experiment results.

## Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/self-supervised-pirl.git
cd self-supervised-pirl
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```


### 3. Download Dataset

The implementation uses STL10 dataset, which can be downloaded from [here](http://ai.stanford.edu/~acoates/stl10/)

Dataset setup steps
```bash
1. Download raw data from above link to ./raw_stl10/
2. Run stl10_data_load.py. This will save three directories train, test and unlabelled in ./stl10_data/
```

### 4. Training and Evaluation Steps (Pre-train with PIRL)

1. Run the following command to pre-train the model using the PIRL methodology:
```bash
python pirl_stl_train_test.py --model-type res18 --batch-size 128 --lr 0.1 --experiment-name exp
```

2. Run script train_stl_after_ssl.py for fine tuning model parameters obtained from self supervised learning, example
```bash
python train_stl_after_ssl.py --model-type res18 --batch-size 128 --lr 0.1  --patience-for-lr-decay 4 --full-fine-tune True --pirl-model-name <relative_model_path from above run>
```

### 5. Results
