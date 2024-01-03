# PIRL-ARI5004-Deep-Learning

# Self-Supervised Learning with PIRL: Reproduction Instructions


This repository contains the implementation of the paper "Self-Supervised Learning of Pretext-Invariant Representations" by Ishan Misra and Laurens van der Maaten. The goal is to reproduce the experiment results presented in the ARI5004-Deep-Learning lecture project report.


## Introduction to PIRL:

PIRL introduces a transformative perspective by emphasizing the importance of semantic representations that remain invariant under image transformations. By focusing on Pretext-Invariant Representation Learning, PIRL ensures that the learned representations retain semantic information despite various transformations. This methodology stands in contrast to traditional approaches that encourage covariance with transformations, making PIRL particularly well-suited for tasks such as object detection and image classification.

<p align="center">
  <img src="https://wiki.math.uwaterloo.ca/statwiki/images/e/ee/SSL_3.JPG" width="500">
</p>


This GitHub repository contains the implementation of PIRL, as described in the paper by Ishan Misra and Laurens van der Maaten. The provided code allows users to pre-train models using the PIRL methodology, fine-tune them for specific tasks, and evaluate the quality of learned representations across various benchmarks. The reproducibility of results and ease of use make this repository a valuable resource for researchers and practitioners interested in self-supervised learning and representation learning.


## Prerequisites

- Python 3.x
- Pytorch 1.13.1+cu116 (installation instructions [here](https://pytorch.org/get-started/locally/))

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


### 2. Download Dataset

The implementation uses STL10 dataset, which can be downloaded from [here](http://ai.stanford.edu/~acoates/stl10/)

Dataset setup steps
```bash
1. Download raw data from above link to ./raw_stl10/
2. Run stl10_data_load.py. This will save three directories train, test and unlabelled in ./stl10_data/
```

### 3. Training and Evaluation Steps (Pre-train with PIRL)

1. Run the following command to pre-train the model using the PIRL methodology:
```bash
python pirl_stl_train_test.py --model-type res18 --batch-size 128 --lr 0.1 --experiment-name exp
```

2. Run script train_stl_after_ssl.py for fine tuning model parameters obtained from self supervised learning, example
```bash
python train_stl_after_ssl.py --model-type res18 --batch-size 128 --lr 0.1  --patience-for-lr-decay 4 --full-fine-tune True --pirl-model-name <relative_model_path from above run>
```


3. Run script train_stl_after_ssl_normal_model.py for fine-tuning model parameters obtained from Resnet-18, example

```bash
python train_stl_after_ssl_normal_model.py --model-type res18 --batch-size 128 --lr 0.1  --patience-for-lr-decay 4 --full-fine-tune True
```

### 4. Results

We present the experiment results along with graphical representations for bet ter visualization.
<br>

<p align="center">
  <img src="https://github.com/erdincsaglamli/PIRL-ARI5004-Deep-Learning/blob/main/grph_1.png?raw=true" width="500">
  
</p>
Graph 1: PIRL Task Loss Graph. The graph illustrates the progress of our model in learning the Jigsaw task across 50 epochs. The PIRL methodology is effective in capturing the semantic information required for the pretext task.


<p align="center">
  <img src="https://github.com/erdincsaglamli/PIRL-ARI5004-Deep-Learning/blob/main/grph_2.png?raw=true" width="500">
  
</p>
Graph 2: Normal ResNet-18 Fine-tuning. After 30 epochs of fine-tuning, the test set results indicate an average loss of 1.9167, corresponding to an accuracy of 30.25. The graph visually represents the convergence and performance of the model during fine-tuning.

<p align="center">
  <img src="https://github.com/erdincsaglamli/PIRL-ARI5004-Deep-Learning/blob/main/grph_3.png?raw=true" width="500">
  
</p>
Graph 3: ResNet-18 + PIRL Fine-tuning. After 30 epochs of fine-tuning with the PIRL pre-trained ResNet-18 back bone, the test set results show an average loss of 1.4884, with an accuracy of
46.25. The comparison with the normal fine-tuning results highlights the improvement achieved by incorporating PIRL pre-training.
<br>

<br>
The fine-tuning results demonstrate the impact of PIRL pre-training on the model’s accuracy and loss during the specific task of object detection. The improved accuracy in the PIRL finetuned model indicates the effectiveness of the self-supervised learning approach in enhancing the representation learning process.

<br>

### 5. References

* Misra, I., van der Maaten, L. Self-Supervised Learning of Pretext-Invariant Representations. In Conference on Computer Vision and Pattern Recognition (CVPR), 2020.
* Noroozi, M., Favaro, P. Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles. In European Conference on Computer Vision (ECCV), 2016.
* Gidaris, S., Komodakis, N. Unsupervised Representation Learning by Predicting Image Rotations. In International Conference on Learning Represen tations (ICLR), 2018.
* Kharitonov, E., Denisov, A., Sattarov, T., Khomenko, D. Unsupervised Object Localization with Frame Rotation Prediction. In International Conference on Learning Representations (ICLR), 2020.
* He, K., Fan, H., Wu, Y., Xie, S., Girshick, R. Momentum Contrast for Unsupervised Visual Representation Learning. In Conference on Computer Vision and Pattern Recognition (CVPR), 2020

