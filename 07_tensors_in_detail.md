# Tensors — What They Are and All Types

## What is a Tensor?

A **Tensor** is a generalized mathematical container for data,
represented as a multi-dimensional array of numbers, with a specific
**rank (number of dimensions)**, **shape**, and **data type**.

> In Deep Learning, EVERYTHING — input data, weights, biases, gradients,
> activations, outputs — is represented as a tensor.
> Tensors are the fundamental data structure of frameworks like
> **TensorFlow** and **PyTorch** (the name "TensorFlow" literally
> comes from tensors flowing through a computation graph).

### Simple Definition
```
A Tensor = A container that can hold data in N dimensions

0D Tensor = a single number
1D Tensor = a list of numbers
2D Tensor = a table of numbers (rows × columns)
3D Tensor = a cube of numbers
ND Tensor = generalization to any number of dimensions
```

> **Analogy:** 📦 Think of tensors as boxes within boxes.
> A scalar is a single item. A vector is a row of items.
> A matrix is a grid of items. A 3D tensor is a stack of grids
> (like pages in a book). It keeps generalizing upward.

---

## Tensor vs Array vs Matrix — Are They the Same?

| Term | Used In | Meaning |
|---|---|---|
| **Array** | Computer Science / Programming | General term for ordered collection of elements |
| **Matrix** | Mathematics (Linear Algebra) | Strictly 2D array (rows × columns) |
| **Tensor** | Physics, ML, Deep Learning | N-dimensional generalization — includes scalar, vector, matrix, and beyond |

> In practice, in deep learning frameworks (PyTorch, TensorFlow),
> "Tensor" is just used as the standard term for ANY array,
> regardless of dimension — scalars and vectors are also called tensors.

---

## Key Properties of a Tensor

| Property | Meaning | Example |
|---|---|---|
| **Rank / Order** | Number of dimensions (axes) | A matrix has rank 2 |
| **Shape** | Size along each dimension | (3, 4) means 3 rows, 4 columns |
| **Axis** | A specific dimension of the tensor | axis=0 is rows, axis=1 is columns |
| **Data Type (dtype)** | Type of values stored | float32, int64, bool |
| **Size / Number of elements** | Total values in the tensor | shape (3,4) → size = 12 |

```python
import numpy as np

t = np.array([[1, 2, 3], [4, 5, 6]])

print(t.ndim)   # 2  → Rank (2D tensor)
print(t.shape)  # (2, 3) → 2 rows, 3 columns
print(t.dtype)  # int64
print(t.size)   # 6  → total elements
```

---
---

## Types of Tensors — By Rank (Dimensionality)

This is the **most important classification** — by number of dimensions (axes).

```
RANK-0 (Scalar)  → Single number
RANK-1 (Vector)  → 1D array
RANK-2 (Matrix)  → 2D array
RANK-3 (3D Tensor) → Cube of numbers
RANK-4 (4D Tensor) → Batch of cubes
RANK-5+ (ND Tensor) → Higher dimensional generalization
```

---

### 🔹 Rank-0 Tensor — Scalar

**Definition:** A single number. Zero dimensions, zero axes.

```python
scalar = tf.constant(42)
# shape: ()
# A scalar has NO axes — just a single value
```

| Property | Value |
|---|---|
| Dimensions | 0 |
| Shape | () |
| Example | 5, 3.14, -2, True |

### Real-Life Examples of Scalars:
| Use Case | Example |
|---|---|
| Temperature reading | 36.6°C |
| Single loss value | loss = 0.0234 |
| A single prediction probability | 0.87 (87% confidence) |
| Learning rate | 0.001 |
| Single pixel intensity (grayscale) | 128 |

---

### 🔹 Rank-1 Tensor — Vector

**Definition:** A 1-dimensional array of numbers — a list/sequence of values.

```python
vector = tf.constant([10, 20, 30, 40])
# shape: (4,)
# One axis with 4 elements
```

| Property | Value |
|---|---|
| Dimensions | 1 |
| Shape | (n,) |
| Example | [1, 2, 3, 4, 5] |

### Real-Life Examples of Vectors:
| Use Case | What the Vector Represents |
|---|---|
| 🏠 House features | [area=1200, rooms=3, age=5, price_per_sqft=4500] |
| 📊 Word embedding | [0.23, -0.45, 0.67, ...] — 300-dim vector representing a word's meaning |
| 🎵 Audio sample | A sequence of amplitude values over time |
| 📈 Stock prices | [100, 102, 98, 105, 110] — closing prices over 5 days |
| 🧬 DNA sequence (encoded) | [0, 1, 1, 0, 2, 3, 1, ...] — encoded nucleotides |
| 🎯 Model output (multi-class) | [0.1, 0.7, 0.2] — probability for each of 3 classes |
| 👤 User profile | [age=25, income=50000, purchases=12] |

---

### 🔹 Rank-2 Tensor — Matrix

**Definition:** A 2-dimensional array — rows and columns, like a spreadsheet table.

```python
matrix = tf.constant([[1, 2, 3],
                       [4, 5, 6]])
# shape: (2, 3)
# 2 rows, 3 columns
```

| Property | Value |
|---|---|
| Dimensions | 2 |
| Shape | (rows, columns) |
| Example | [[1,2],[3,4]] |

### Real-Life Examples of Matrices:
| Use Case | What the Matrix Represents |
|---|---|
| 📊 Tabular dataset | Rows = samples, Columns = features (e.g., 1000 customers × 10 features) |
| 🖼️ Grayscale image | Rows × Columns = pixel intensities (height × width) |
| 🔢 Weight matrix in Neural Network | Connects one layer to another — shape (input_neurons, output_neurons) |
| 📈 Time series (multiple sensors) | Rows = time steps, Columns = sensor readings |
| 🗣️ Word co-occurrence matrix (NLP) | Rows/Columns = words, values = co-occurrence count |
| 🎮 Game board state | Chess board = 8×8 matrix representing piece positions |
| 🌍 Distance matrix | Rows/Columns = cities, values = distance between them |

---

### 🔹 Rank-3 Tensor — 3D Tensor

**Definition:** A 3-dimensional array — like a "cube" or stack of matrices.

```python
tensor3d = tf.constant([[[1,2],[3,4]],
                          [[5,6],[7,8]],
                          [[9,10],[11,12]]])
# shape: (3, 2, 2)
# 3 matrices, each of shape (2,2)
```

| Property | Value |
|---|---|
| Dimensions | 3 |
| Shape | (depth, rows, columns) |
| Example | A stack of matrices |

### Real-Life Examples of 3D Tensors:
| Use Case | What the 3D Tensor Represents |
|---|---|
| 🖼️ Color image (RGB) | (Height, Width, Channels) → e.g., (224, 224, 3) for a 224×224 RGB image |
| 📈 Time series with multiple features | (Time steps, Samples, Features) |
| 📝 Batch of text sequences | (Batch size, Sequence length, Embedding dim) — one sentence's word embeddings |
| 🎞️ Grayscale video | (Frames, Height, Width) |
| 🧬 Medical imaging (single slice + channels) | (Height, Width, Modalities) — e.g., MRI with multiple scan types |
| 🗣️ Audio spectrogram | (Time, Frequency, Channels) |

---

### 🔹 Rank-4 Tensor — 4D Tensor

**Definition:** A 4-dimensional array — most commonly used for **batches of images**
in deep learning.

```python
tensor4d = tf.random.normal((32, 224, 224, 3))
# shape: (32, 224, 224, 3)
# 32 images, each 224x224 pixels, with 3 color channels (RGB)
```

| Property | Value |
|---|---|
| Dimensions | 4 |
| Shape | (Batch, Height, Width, Channels) |
| Example | A batch of RGB images |

### Real-Life Examples of 4D Tensors:
| Use Case | What the 4D Tensor Represents |
|---|---|
| 🖼️ Batch of color images (CNN input) | (Batch_size, Height, Width, Channels) — standard CNN input format |
| 🎞️ Color video (single video) | (Frames, Height, Width, Channels) |
| 🩻 Batch of medical scans | (Batch, Height, Width, Modalities) |
| 📊 Batch of multi-channel time series | (Batch, Time, Features, Channels) |
| 🎮 Game state history | (Batch, Frame_stack, Height, Width) — used in RL (e.g., Atari DQN) |

> **Note on formats:**
> - TensorFlow default: **NHWC** = (Batch, Height, Width, Channels)
> - PyTorch default: **NCHW** = (Batch, Channels, Height, Width)

---

### 🔹 Rank-5 Tensor — 5D Tensor

**Definition:** A 5-dimensional array — typically used for **batches of videos**.

```python
tensor5d = tf.random.normal((16, 30, 224, 224, 3))
# shape: (16, 30, 224, 224, 3)
# 16 videos, each with 30 frames, 224x224 resolution, RGB channels
```

| Property | Value |
|---|---|
| Dimensions | 5 |
| Shape | (Batch, Frames, Height, Width, Channels) |
| Example | A batch of color videos |

### Real-Life Examples of 5D Tensors:
| Use Case | What the 5D Tensor Represents |
|---|---|
| 🎬 Batch of videos (Video Classification) | (Batch, Frames, Height, Width, Channels) |
| 🧠 3D Medical Scans (batch) | (Batch, Depth/Slices, Height, Width, Channels) — e.g., 3D MRI volumes |
| 🎮 Sequential game frames (batch, RL) | (Batch, Time, Height, Width, Channels) |
| 🛰️ Satellite imagery time series | (Batch, Time, Height, Width, Spectral_bands) |

---

### 🔹 Rank-N — Higher Dimensional Tensors

Beyond rank 5, tensors are used in specialized scientific and
research applications.

| Use Case | Dimensions |
|---|---|
| Multi-modal video + audio + text batch | 6D+ combining multiple sequential modalities |
| Physics simulations (spacetime tensors) | 4D spacetime + multiple physical properties |
| Hyperspectral satellite data over time | (Batch, Time, Height, Width, Spectral_bands, Sensors) |
| Quantum computing state tensors | Can have very high rank depending on qubit count |

> In practice, most real-world deep learning rarely goes beyond rank-5.
> Higher-rank tensors appear mainly in physics, scientific computing,
> and specialized multi-modal research.

---
---

## Visual Summary — Tensor Ranks

```
RANK 0 (Scalar)        5
                        ────────────────

RANK 1 (Vector)         [5, 3, 8, 1]
                        ────────────────

RANK 2 (Matrix)         [5, 3, 8]
                         [1, 9, 2]
                         [7, 4, 6]
                        ────────────────

RANK 3 (3D Tensor)      Stack of matrices
                         ▢▢▢
                        ▢▢▢
                       ▢▢▢
                        ────────────────

RANK 4 (4D Tensor)      Batch of stacked matrices
                         [▢▢▢] [▢▢▢] [▢▢▢] [▢▢▢]
                        ────────────────

RANK 5 (5D Tensor)      Batch of sequences of stacked matrices
                         [[▢▢▢][▢▢▢]] [[▢▢▢][▢▢▢]] ...
```

---
---

## Types of Tensors — By Data Type (dtype)

Tensors also vary by the **type of data** they store:

| Data Type | Description | Example | Common Use |
|---|---|---|---|
| **float32** | 32-bit floating point | 3.14159 | Standard for most DL training (weights, activations) |
| **float64** | 64-bit floating point (double precision) | 3.14159265358979 | High-precision scientific computing |
| **float16 / bfloat16** | 16-bit floating point (half precision) | 3.14 | Faster training, mixed precision, memory savings |
| **int8 / int16 / int32 / int64** | Integer values | 1, 2, 3, -5 | Indices, labels, counts, quantized models |
| **bool** | True / False values | True, False | Masks, conditions, binary flags |
| **string / unicode** | Text data | "hello" | NLP raw text tensors (TensorFlow supports string tensors) |
| **complex64 / complex128** | Complex numbers (real + imaginary) | 3+4j | Signal processing, FFT operations |

### Why Data Type Matters:
```
- float32 is the DEFAULT for most deep learning (balance of precision and speed)
- float16/bfloat16 used in MIXED PRECISION TRAINING
  → 2x faster training, half the memory, slight precision loss
- int8 used in QUANTIZED MODELS
  → For deploying models on mobile/edge devices (smaller, faster)
- bool tensors used for MASKING
  → e.g., attention masks in Transformers (which tokens to attend to)
```

---
---

## Types of Tensors — By Special Properties / Structure

Beyond rank and dtype, tensors can also be categorized by **mathematical
structure** or **role** in computation:

### 🔹 Constant Tensor

A tensor whose value **cannot change** after creation. Used for fixed data.

```python
const = tf.constant([1, 2, 3])
```
**Use case:** Storing fixed input data, hyperparameters that don't change.

---

### 🔹 Variable Tensor

A tensor whose value **CAN change** — used to store and update
**trainable parameters** (weights, biases) during training.

```python
weight = tf.Variable([0.5, 0.3, 0.8])
```
**Use case:** Neural network weights and biases — updated via gradient descent
during every training step.

---

### 🔹 Placeholder Tensor (older TensorFlow 1.x concept)

A tensor that acts as an **empty slot** — data is fed into it later
during execution. (Mostly deprecated in TensorFlow 2.x / eager execution,
but conceptually important to understand.)

**Use case:** Defining computation graphs in TF1.x before data was available.

---

### 🔹 Sparse Tensor

A tensor where **most values are zero**, stored efficiently by only
recording the non-zero values and their positions (instead of storing
every single zero).

```python
# Instead of storing: [0, 0, 0, 5, 0, 0, 8, 0]
# Sparse representation stores only:
# indices: [3, 6], values: [5, 8], shape: [8]
```

| Property | Description |
|---|---|
| **Storage** | Only non-zero elements + their indices |
| **Memory savings** | Massive for highly sparse data |

### Real-Life Examples of Sparse Tensors:
| Use Case | Why Sparse |
|---|---|
| 📝 One-hot encoded text (Bag of Words) | Vocabulary of 50,000 words, but each document uses only ~100 — 99.8% zeros |
| 🛒 User-item rating matrix (recommendation) | Millions of users × millions of products, but each user rates very few — mostly zeros |
| 🌐 Graph adjacency matrix | Most nodes aren't connected to most other nodes |
| 🧬 Genomics data | Most gene expressions are zero/inactive in a given sample |
| 🔢 One-hot encoded categorical features | Only one "1" among thousands of "0"s |

---

### 🔹 Dense Tensor

A tensor where **most or all values are non-zero**, stored normally
(every value explicitly stored). This is the standard/default tensor type.

**Use case:** Image pixels, normal numerical feature data, weight matrices.

---

### 🔹 Ragged Tensor

A tensor where **dimensions can have variable length** —
used when data is NOT uniformly shaped (unlike normal tensors which
require rectangular/uniform shape).

```python
ragged = tf.ragged.constant([[1, 2, 3], [4, 5], [6]])
# Row 1 has 3 elements, Row 2 has 2, Row 3 has 1
# A normal tensor REQUIRES all rows to be the same length
```

### Real-Life Examples of Ragged Tensors:
| Use Case | Why Ragged |
|---|---|
| 📝 Sentences of different lengths (NLP) | "I am happy" (3 words) vs "I am extremely happy today" (5 words) |
| 💬 Chat conversation history | Different users have different numbers of messages |
| 🎬 Variable-length video clips | Videos have different durations/frame counts |
| 🌳 Tree-structured data | Each node can have a different number of children |
| 🛒 Customer purchase history | Each customer made a different number of purchases |

> **Common solution:** Padding — make all sequences the same length
> by adding zeros — converts ragged → dense (used commonly in NLP/RNNs).

---

### 🔹 One-Hot Tensor

A special tensor where each row has exactly **one "1"** and the
rest are "0" — used to represent categorical labels.

```python
# Representing classes: Cat=0, Dog=1, Bird=2
one_hot = tf.one_hot([1, 0, 2], depth=3)
# [[0,1,0],
#  [1,0,0],
#  [0,0,1]]
```

**Use case:** Encoding categorical labels for classification tasks,
encoding categorical features.

---

### 🔹 Embedding Tensor

A dense, learned tensor that represents discrete items (words, users,
products) as continuous vectors capturing semantic meaning/relationships.

```python
# Word "king" might be represented as:
embedding = [0.21, -0.45, 0.67, 0.12, ...]  # e.g., 300 dimensions
```

### Real-Life Examples of Embedding Tensors:
| Use Case | What It Represents |
|---|---|
| 🗣️ Word Embeddings (Word2Vec, GloVe, BERT) | Words → dense vectors capturing meaning |
| 🎬 Movie Embeddings (Recommendation) | Movies → vectors capturing genre/style similarity |
| 👤 User Embeddings | Users → vectors capturing preferences |
| 🖼️ Image Embeddings (CNN features) | Images → vectors capturing visual content |
| 🧬 Gene Embeddings | Genes → vectors capturing biological function similarity |

---
---

## Tensor Operations You Should Know

| Operation | Description | Example |
|---|---|---|
| **Reshape** | Change shape without changing data | (6,) → (2,3) |
| **Transpose** | Swap axes | (3,4) → (4,3) |
| **Slicing** | Extract part of a tensor | tensor[0:2, :] |
| **Broadcasting** | Operate on tensors of different shapes automatically | (3,4) + (4,) → (3,4) |
| **Concatenation** | Join tensors along an axis | Combine two (3,4) → (6,4) |
| **Matrix Multiplication** | Core operation in neural networks | (m,n) × (n,p) = (m,p) |
| **Element-wise Operations** | Add, subtract, multiply element by element | [1,2]+[3,4] = [4,6] |
| **Reduction** | Sum, mean, max along an axis | sum(axis=0) |
| **Squeeze / Unsqueeze** | Remove/add dimensions of size 1 | (1,3,1) → (3,) |

---
---

## Why Tensors Are Central to Deep Learning

```
INPUT DATA          → Tensor (e.g., batch of images: 4D)
        │
        ▼
WEIGHTS & BIASES     → Tensors (learnable parameters)
        │
        ▼
FORWARD PASS         → Tensor operations (matrix multiplication,
                        convolution — all tensor math)
        │
        ▼
ACTIVATIONS          → Tensor (output of each layer)
        │
        ▼
LOSS                 → Scalar Tensor (single number to minimize)
        │
        ▼
GRADIENTS            → Tensors (same shape as weights —
                        tell us how to update each weight)
        │
        ▼
UPDATED WEIGHTS       → Tensors (updated via gradient descent)
```

> This is why frameworks are literally named after this concept:
> **TensorFlow** = "Tensors flowing through a computational graph"
> **PyTorch** = "Torch" library + Python, also fundamentally tensor-based

---
---

## Summary Table — All Tensor Types

### By Rank (Dimensionality):

| Rank | Name | Shape Example | Real-World Example |
|---|---|---|---|
| 0 | Scalar | () | Single loss value: 0.023 |
| 1 | Vector | (n,) | Word embedding, feature vector |
| 2 | Matrix | (m, n) | Tabular dataset, grayscale image |
| 3 | 3D Tensor | (d, h, w) | RGB image, text sequence batch |
| 4 | 4D Tensor | (b, h, w, c) | Batch of RGB images (CNN input) |
| 5 | 5D Tensor | (b, t, h, w, c) | Batch of videos |
| N | ND Tensor | (...) | Scientific simulations, multi-modal data |

### By Data Type:

| dtype | Precision | Common Use |
|---|---|---|
| float32 | Standard | Default for training |
| float16/bfloat16 | Half | Mixed precision training |
| int8/int32/int64 | Integer | Labels, indices, quantization |
| bool | Binary | Masks, conditions |
| string | Text | NLP raw text |

### By Structure:

| Type | Key Property | Use Case |
|---|---|---|
| Constant | Fixed value | Static input data |
| Variable | Trainable | Model weights/biases |
| Sparse | Mostly zeros, memory-efficient | Text BoW, recommendation matrices |
| Dense | Mostly non-zero | Images, standard features |
| Ragged | Variable-length dimensions | Sentences, variable sequences |
| One-Hot | Single "1" per row | Categorical labels |
| Embedding | Dense, learned representation | Word/item embeddings |

---

*Next Topic → Neural Networks Fundamentals (Perceptron, Activation Functions, Forward/Backward Propagation)*
