# Rig Inversion by Training a Differentiable Rig Function

This code serves an an example of using the technique described in the paper Rig Inversion by Training a Differentiable Rig Function published at Siggraph Asia 2022.

[Rig Inversion by Training a Differentiable Rig Function](https://arxiv.org/abs/2301.09567)

Rig inversion is demonstrated using a toy rig.

## How to use

### Step 1 : Generate a training dataset using your rig

python generate_toy_dataset.py

### Step 2 : Train a model to approximate the rig and test is using animations

python train_rig_approximation.py

### Step 3 : Inverse the rig using the rig approximation trained in step 2

python inverse_rig.py

## Contents

- `generate_toy_dataset.py`: Will generate a dataset to train the rig approximation of our toy rig
- `inverse_rig.py`: Inverse the rig for the test mesh data using a trained rig approximation
- `model.py`: Model definition for rig approximation and rig inversion
- `rig.py`: Definition of our toy rig function
- `train_rig_approximation.py`: Trains a rig approximation using a dataset of rig function data points

## References

If you use this technique, please cite the paper:  

> *Marquis Bolduc, Mathieu and Phan, Hau Nghiep*. **[Rig Inversion by Training a Differentiable Rig Function](https://arxiv.org/abs/2301.09567)**. SIGGRAPH Asia 2022 Technical Communications.

BibTeX:

```
@inproceedings{10.1145/3550340.3564218,
author = {Marquis Bolduc, Mathieu and Phan, Hau Nghiep},
title = {Rig Inversion by Training a Differentiable Rig Function},
year = {2022},
isbn = {9781450394659},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3550340.3564218},
doi = {10.1145/3550340.3564218},
abstract = {Rig inversion is the problem of creating a method that can find the rig parameter vector that best approximates a given input mesh. In this paper we propose to solve this problem by first obtaining a differentiable rig function by training a multi layer perceptron to approximate the rig function. This differentiable rig function can then be used to train a deep learning model of rig inversion.},
booktitle = {SIGGRAPH Asia 2022 Technical Communications},
articleno = {15},
numpages = {4},
keywords = {computer animation, neural networks, rig inversion},
location = {Daegu, Republic of Korea},
series = {SA '22}
}
```

## Authors

<p align="center"><a href="https://seed.ea.com"><img src="logo/SEED.jpg" width="150px"></a><br>
<b>Search for Extraordinary Experiences Division (SEED) - Electronic Arts <br> http://seed.ea.com</b><br>
We are a cross-disciplinary team within EA Worldwide Studios.<br>
Our mission is to explore, build and help define the future of interactive entertainment.</p>

This technique was created by Mathieu Marquis Bolduc and Hau Nghiep Phan

## Licenses

- The source code uses *BSD 3-Clause License* as detailed in [LICENSE.md](LICENSE.md)
