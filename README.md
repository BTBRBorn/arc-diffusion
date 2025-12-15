# 🤖 arc-diffusion: From-Scratch Masked Diffusion Model for the Abstraction and Reasoning Corpus

## 🌟 Overview

This repository presents a complete, **from-scratch** implementation of a **Masked Diffusion Model** pipeline specifically designed to tackle the **Abstraction and Reasoning Corpus (ARC)** challenge.

The ARC challenge requires an agent to generalize abstract concepts from very few examples and apply that reasoning to novel tasks. This project approaches ARC as a sequence-to-sequence prediction problem, where the input and output grids are serialized into tokens and processed by a custom-built masked diffusion model, leveraging the power of attention mechanisms for complex pattern recognition.

### Key Features ✨

  * **Custom Implementation:** A complete, modular implementation of a Masked Diffusion architecture (`model.py`), inspiration for the model comes from this paper: https://arxiv.org/pdf/2502.09992.
  * **ARC Data Pipeline:** Scripts to generate, tokenize, mask and augment training data from the original ARC tasks (`create_training_data.py`, `get_dataloaders.py`).
  * **Custom Tokenizer & Serialization:** Specialized tokenizer (`get_tokenizer.py`) designed to convert 2D ARC grids (10 colors + grid structure) into a linear sequence of tokens suitable for masked diffusion modeling.
  * **Modular Training Engine:** Separate scripts for training, configuration, and evaluation (`train.py`, `engine.py`, `configurations.py`).

### Missing Features :disappointed:
  * **Inference:** Inference code is not complete. `eval.py` still needs work.

### Purpose

This project is dedicated to exploring model architectures that are both more efficient and possess stronger innate reasoning capabilities.

This repository provides a flexible training and evaluation pipeline for testing various architectural hypotheses using ARC-AGI-2 data. The public version includes a masked diffusion model with multi-head attention, mlp layer with gelu activations and layer normalization in `model.py`, which serves as a primary baseline.

The pipeline is fundamentally designed to evaluate **any masked diffusion model**. By modifying `model.py`, it can be easily repurposed to test entirely different architectures and concepts.