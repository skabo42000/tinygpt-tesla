# Glossary

Plain-English meanings of every term we meet. It grows with each lesson.

## Shape letters (used in every lesson)

| Letter | Means | Example size |
|---|---|---|
| **B** | Batch: how many text chunks we process at once | 4 |
| **T** | Time: position inside a chunk (how many characters of context) | 8 |
| **C** | Channels: how many numbers describe each character (embedding size) | 32 |
| **V** | Vocab size: how many different tokens exist | 65 |
| **hs** | Head size: width of one attention head | 16 |

## Lesson 0: Setup

- **PyTorch:** a Python library that does fast math on big grids of numbers and can work out how to improve a model automatically.
- **CPU:** the laptop's main processor. A few strong, general-purpose cores.
- **GPU / CUDA:** an NVIDIA graphics card, and the software PyTorch uses to run on it. It has thousands of small cores, which makes it great for the repeated multiplications neural networks need. This laptop doesn't have one, so we train on the CPU.
- **Thread:** a parallel lane of work. PyTorch splits a big calculation across several threads.
- **Matrix multiplication (`@`):** multiplying rows of one grid of numbers against columns of another. It's the core operation inside a GPT.
- **Virtual environment:** a private folder holding this project's Python and libraries, so other projects can't break it. Ours is at `C:\Users\bosko\.venvs\llm-project`.
- **Kernel (in a notebook):** the Python process that runs your notebook cells. It must point at our virtual environment.
- **Dataset:** the text the model learns from. For us that's *The Inventions, Researches and Writings of Nikola Tesla* (1894), about 977,000 characters, saved as `data/input.txt`.
- **Boilerplate:** standard text that isn't part of the actual content, such as Project Gutenberg's license at the top and bottom of every book. We strip it out so the model doesn't learn it.

## Lesson 1: Tensors & shapes

- **Tensor:** a grid of numbers with any number of dimensions: a single number, a list, a table, or a stack of tables.
- **Shape:** the size along each dimension, e.g. `(2, 3, 4)`. Read from left (outermost) to right (innermost).
- **ndim:** how many dimensions a tensor has (the length of its shape).
- **Indexing:** picking parts of a tensor. A number removes that dimension; `:` keeps all of it; `-1` means the last one.
- **Broadcasting:** combining tensors of different shapes. Shapes are lined up from the right; each pair must be equal or one of them 1 (which gets stretched).
- **dim (in sum/mean):** which dimension to add up or average along. That dimension disappears (or stays as size 1 with `keepdim=True`).
- **Matrix multiply rule:** (n, k) @ (k, m) → (n, m). The inner sizes must match and disappear. With 3-D tensors, only the last two dimensions are multiplied; the rest are treated as a batch.
- **Transpose (`.T`, `.transpose(-2, -1)`):** swap rows and columns.
- **`.view`:** regroup the same numbers into a new shape without moving them, e.g. (B, T, C) → (B·T, C).
