# nn-lpf
 
A collection of low-pass filters (LPF) for antialiasing neural network architectures. Low-pass filtering before downsampling improves both classification accuracy and shift robustness of convolutional neural networks (CNNs). This project was developed as part of a bachelor thesis and is intended to support further research in antialiased networks.
 
## Overview
 
Standard CNN downsampling operations (strided convolutions, max-pooling) violate the Nyquist–Shannon sampling theorem, which causes aliasing artifacts and makes predictions sensitive to small spatial shifts in the input. This repository provides drop-in antialiasing modules that can be inserted into existing architectures to address this problem.

## **other informations coming soon**

## Reproducibility
 
A fixed random seed (`--seed 1`) is used by default. Deterministic CUDA operations are enabled for full reproducibility, though this may reduce training speed.
 
## Acknowledgments
 
This project builds on ideas from:
 
- Zhang, R. (2019). *Making Convolutional Networks Shift-Invariant Again.* ICML.
- Hendrycks, D. & Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.* ICLR.
 
## License
 
This project was developed for academic research purposes as part of a bachelor thesis.
