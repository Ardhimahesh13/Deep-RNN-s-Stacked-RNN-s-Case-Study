# Deep Recurrent Neural Networks (Stacked RNNs)

[![Deep Learning](https://img.shields.io/badge/Deep_Learning-RNNs-blue?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.x-yellow?logo=python)](https://www.python.org/)

This repository contains notes and implementation details for **Deep RNNs (Stacked RNNs, LSTMs, and GRUs)**. It explores how stacking recurrent layers vertically increases the representation power of the network, allowing it to solve complex tasks like Machine Translation and Speech Recognition.

> **Reference:** [CampusX - Deep RNNs | Stacked RNNs | Stacked LSTMs | Stacked GRUs](https://youtu.be/mlDkTrlLaio?list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn)

## 📌 Table of Contents
- [Overview](#overview)
- [Architecture & Intuition](#architecture--intuition)
- [Mathematical Notation](#mathematical-notation)
- [Implementation (Keras)](#implementation-keras)
- [Parameter Calculation](#parameter-calculation)
- [Pros & Cons](#pros--cons)

## 🧠 Overview

A **Deep RNN** (or Stacked RNN) involves stacking multiple RNN layers on top of each other. 
- Just as adding hidden layers to an ANN improves its ability to capture complex non-linear patterns, adding recurrent layers allows an RNN to learn **hierarchical representations** of sequential data.
- **Lower Layers:** Capture primitive features (e.g., individual words).
- **Higher Layers:** Capture abstract concepts (e.g., sentence sentiment or context).

## 🏗 Architecture & Intuition

The architecture can be visualized as a **2D Grid**:
1.  **Horizontal Axis (Time $t$):** Information flows from $t-1$ to $t$.
2.  **Vertical Axis (Depth $l$):** Information flows from layer $l-1$ to layer $l$.



### The Flow
In a standard RNN, the input comes only from the dataset. In a Deep RNN:
* **Layer 1:** Receives raw input ($X_t$).
* **Layer 2:** Receives the **hidden state** of Layer 1 as its input.
* **Layer $N$:** Receives the hidden state of Layer $N-1$.

## Pp Mathematical Notation

For a specific cell at layer $l$ and time $t$, the hidden state $h^l_t$ is calculated using inputs from the previous time step ($h^l_{t-1}$) and the previous layer ($h^{l-1}_t$).

$$h^l_t = \tanh(W^l \cdot h^l_{t-1} + U^l \cdot h^{l-1}_t + b^l)$$

Where:
* $W^l$: Weights for the recurrent connection (time axis).
* $U^l$: Weights for the input connection (depth axis).
* $b^l$: Bias.

## 💻 Implementation (Keras)

The most critical aspect of implementing Stacked RNNs in Keras is the `return_sequences` argument.

* **Intermediate Layers:** Must set `return_sequences=True`. This ensures the layer outputs a sequence (3D tensor) rather than a single vector, allowing the next RNN layer to process it.
* **Final Layer:** Typically sets `return_sequences=False` (unless connecting to another sequence model like Attention).

### Code Snippet

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Embedding

model = Sequential()

# Embedding Layer
model.add(Embedding(input_dim=10000, output_dim=32))

# Layer 1: MUST return sequences to feed Layer 2
model.add(SimpleRNN(units=5, return_sequences=True)) 

# Layer 2: Feeds into a Dense layer, so we can return just the final state
model.add(SimpleRNN(units=5, return_sequences=False))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

model.summary()


## 🧮 Parameter Calculation
<img width="737" height="575" alt="image" src="https://github.com/user-attachments/assets/137f61bd-2e93-4b36-82ef-e9ab249e9769" />

