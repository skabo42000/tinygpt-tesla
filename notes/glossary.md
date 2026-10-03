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

## Lesson 2: How a model learns

- **Parameter:** a number inside the model that training adjusts (like `w` and `b`). Our GPT will have hundreds of thousands.
- **Loss:** one number measuring how wrong the model is. Lower is better; 0 is perfect.
- **Mean squared error:** a loss for number predictions: the average of (prediction − truth)².
- **Gradient:** for each parameter, which direction (and how strongly) nudging it up would change the loss. We move against it.
- **`loss.backward()`:** makes PyTorch work out the gradient of every parameter automatically.
- **Learning rate:** the step size of each nudge. Too small is slow; too big overshoots and blows up.
- **Optimizer:** the PyTorch tool that applies the nudges to all parameters (`SGD` today, `AdamW` for the GPT).
- **`nn.Linear`:** a ready-made "multiply by weights, add bias" layer that holds its own parameters.
- **The training loop:** forward → loss → zero_grad → backward → step, repeated many times.

## Lesson 3: Scores → probabilities → loss

- **Logits:** the model's raw scores, one per possible next character. They can be any number.
- **Softmax:** turns scores into probabilities that are all positive and add up to 1. Higher scores get a much bigger share.
- **Cross-entropy loss:** −ln(probability given to the correct answer). 0 when fully sure and right; large when confident and wrong.
- **ln(V) baseline:** the loss of a model that knows nothing and gives all V characters equal odds. For our 85 characters it's ln(85) ≈ 4.44, the loss we expect at the very start of training.
- **Embedding (`nn.Embedding`):** a lookup table with one row of learnable numbers per character.
- **Sampling (`torch.multinomial`):** picking a character at random, weighted by its probability, like a weighted dice roll. It gives the model variety instead of always choosing the top guess.

## Lesson 4: Tokenizer

- **Token:** one unit of text the model reads or writes. For us, one character.
- **Token id:** the number that stands for a token (e.g. `T` = 47).
- **Tokenizer:** the two-way dictionary between text and token ids. **Encode** = text → ids; **decode** = ids → text.
- **Vocabulary (V):** all the tokens the model knows. Ours: the 85 characters in Tesla's book. The model can never read or write anything outside it.
- **Lossless:** decode(encode(text)) gives back exactly the same text.
- **`stoi` / `itos`:** "string to integer" and "integer to string", the two lookup tables.
- **Frequency-only baseline:** the loss of a model that only knows how common each character is: 3.05 on Tesla's text.

## Lesson 5: Batches

- **Chunk:** a short piece of the book the model reads at once.
- **block_size (T):** the chunk length, and the most text the model can ever look back at (its **context**).
- **Targets (y):** the chunk shifted by one character. At every position, the "right answer" is simply the next character.
- **Batch (B):** several random chunks stacked into a (B, T) grid, so the model learns from many places at once.
- **Training / validation split:** 90% of the book to learn from, 10% held back as an "exam" the model never trains on. If the training loss keeps falling but the validation loss rises, the model is memorizing instead of learning (**overfitting**).
- **`get_batch`:** the function that cuts random chunks and their targets out of the book for every training step.

## Lesson 6: Bigram model

- **Language model:** a model that gives probabilities for the next token, and can write text by repeatedly sampling one.
- **Bigram model:** predicts the next character from only the current one. An 85 × 85 table of scores.
- **`nn.Module` / class:** how PyTorch bundles a model: its parameters (`__init__`) plus how to compute the output (`forward`).
- **`generate`:** write text one token at a time: scores at the last position → softmax → sample → append → repeat.
- **AdamW:** the optimizer used for GPTs. It adapts the step size for each parameter separately.
- **`estimate_loss`:** averages the loss over many batches (train and val) for a steady reading, without learning (`eval()`, `no_grad`).
- **Initialization:** the random starting values of the parameters. Random scores make a fresh model slightly "confidently wrong", so its first loss is a bit above ln(V).

## Lesson 7: The averaging trick

- **Causal / no peeking:** each position may only use itself and earlier positions, never later ones, because when writing text the future doesn't exist yet.
- **Lower-triangular matrix (`torch.tril`):** ones on and below the diagonal, zeros above. Row t marks which positions t is allowed to see.
- **Weights matrix (wei):** a (T, T) table; row t says how much of each position to mix into position t. Each row adds up to 1.
- **Masking (`masked_fill` with −∞):** setting future scores to minus infinity so softmax gives them exactly 0 weight.
- **Weighted sum (`wei @ x`):** mixes information from the allowed positions in one matrix multiply. The heart of attention.

## Lesson 8: Self-attention head

- **Self-attention:** each character decides how much every earlier character matters to it, then takes a weighted mix of their information.
- **Query (q):** "what am I looking for?" One vector per character.
- **Key (k):** "what do I contain?" Compared with queries by dot product: a good match gives a high score.
- **Value (v):** "what do I share if you pick me?" The information that actually gets mixed.
- **Head:** one complete query/key/value attention unit. Lesson 9 uses several side by side.
- **head_size (hs):** the length of each q, k, v vector.
- **Scaling by 1/√hs:** keeps attention scores at a spread of about 1, so softmax stays soft (spread out) at the start instead of locking onto one character.
- **Position embedding:** a learnable vector per position (0..T−1), added to each character's vector so attention knows the order of characters.
- **`register_buffer`:** stores a fixed tensor (like the triangle mask) inside the model without training it.
- **Context window:** the most characters the model can look back at (block_size, now 32). `generate` cuts the input to that length.

## Lesson 9: Multi-head attention + feed-forward

- **Multi-head attention:** several smaller heads run side by side, each learning its own kind of search; their outputs are joined (`torch.cat`) back to the full width.
- **Feed-forward layer:** a small network (widen ×4 → ReLU → shrink back) applied to each character separately, to process what attention gathered.
- **Communication vs computation:** attention moves information *between* characters; feed-forward *thinks about it* within each character.
- **ReLU:** keeps positive numbers, turns negatives into 0. The "bend" that lets a network make decisions; without it, stacked Linear layers collapse into one.
- **Non-linearity / activation function:** the general name for bends like ReLU (GPT-2 uses a smoother one called GELU).
- **`nn.ModuleList` / `nn.Sequential`:** PyTorch containers for a list of layers, or layers run one after another.

## Lesson 10: The full GPT

- **Block:** one round of "talk, then think": `x = x + attention(LayerNorm(x))`, then `x = x + feedforward(LayerNorm(x))`. A GPT stacks several.
- **Residual connection (`x = x + layer(x)`):** each layer adds a correction instead of replacing the information. The original always passes straight through, giving the learning signal a "highway" back through deep stacks.
- **LayerNorm:** re-centers each character's numbers to average 0 and spread 1 (plus two learnable knobs). Keeps numbers healthy through many layers.
- **Pre-norm:** applying LayerNorm *before* each sub-layer, as GPT-2 does.
- **Dropout:** during training, randomly zeroes a fraction of numbers (and scales up the rest), so the model can't rely on any single one. Switched off in `eval` mode.
- **Projection (`proj`):** a Linear layer that mixes the joined heads' outputs before adding them back to the residual path.
- **n_layer / n_head / n_embd:** how many blocks, heads per block, and numbers per character. The main size knobs of a GPT.
- **Overfitting gap:** the difference between training and validation loss. A growing gap means the model is starting to memorize.

## Lesson 11: Package + tests

- **Package:** code organized in files (`src/tinygpt/`) that notebooks and scripts import, so there's one tested version instead of copies.
- **`GPTConfig`:** all the model's size settings in one place (V, T, C, heads, layers, dropout).
- **Parameter budget:** most parameters sit in the feed-forward layers (~62%) and attention q/k/v (~23%). Roughly 12 × C² per block.
- **Test (pytest):** a small function that checks one thing and fails loudly if it breaks. Run all of them with `uv run pytest`.
- **Causality test ("no peeking"):** change one token and check that predictions *before* it don't move at all.
- **Silent bug:** a bug that doesn't crash and may even make numbers look better (like a broken mask). Only a targeted test catches it.
- **Overfit-one-batch check:** if the model can't memorize a single batch, something is wired wrong.

## Lesson 12: Training run + controlling the writing

- **Training script:** training as a program run from the terminal (`uv run python -m tinygpt.train`) instead of a notebook cell, so long runs are safe and repeatable.
- **Checkpoint:** a file holding the model's numbers (plus optimizer state, settings, vocabulary and history), saved during training. `tesla.pt` = latest, `tesla_best.pt` = lowest validation loss.
- **Resume:** continue training from the latest checkpoint after stopping (`--resume`).
- **Step:** one batch through the 5-step training loop. Our run: 3,000 steps × 32 chunks × 128 characters ≈ 12 million characters seen, about 14 passes over the training text.
- **Temperature:** divide the scores by T before softmax. Below 1: safer and more repetitive; above 1: more adventurous with more mistakes. It only matters where the model is unsure.
- **Top-k:** only the k most likely tokens may be picked; the rest are set to −∞. Stops rare bad picks from derailing the text.
- **Prompt:** the starting text the model continues from.

## Lesson 13: Experiments

- **Ablation:** removing one part of a model, keeping everything else identical, and measuring the difference. Shows what each part is worth.
- **Controlled experiment:** same model size, seed, data and steps; only one thing changes.
- **Overfitting:** training loss keeps falling while validation loss turns around and rises. The model memorizes instead of learning general patterns.
- **Early stopping:** keeping the checkpoint with the lowest validation loss (our `tesla_best.pt`), not the last one.
- **Data leakage / cheating:** the model gets access to the answer during training (like a missing mask). Loss looks amazing, real use is terrible.
- **Regularization:** anything that discourages memorizing, such as dropout or more data.

## Lesson 14: BPE tokenizer

- **Byte:** a number from 0 to 255. All text is stored as bytes.
- **UTF-8:** the standard that turns characters into bytes. English letters = 1 byte; ć = 2; Cyrillic = 2 per letter; ⚡ = 3.
- **Byte-Pair Encoding (BPE):** start from the 256 bytes and repeatedly merge the most common neighbouring pair into a new token. Used by GPT-2, GPT-4 and Claude-style tokenizers.
- **Merge:** one learned rule, e.g. `' t' + 'he' → ' the'`. Applied in the order learned when encoding.
- **Pre-split pattern (regex):** splits text into word-like chunks first (word + its leading space, numbers, punctuation) so merges never cross word boundaries.
- **Compression ratio:** characters per token. Ours: 2.22 on Tesla's book with 512 tokens; GPT-4's tokenizer reads whole words.
- **`tiktoken`:** OpenAI's tokenizer library, used here to compare with GPT-2 (50,257 tokens) and GPT-4 (`cl100k_base`, ~100,000 tokens).

## Lesson 15: Characters vs word pieces

- **Bits per character (BPC):** total loss over a text ÷ (number of characters × ln 2). A fair score for models with different tokenizers. Roughly: yes/no questions needed per character. Lower is better.
- **Nats vs bits:** our loss uses the natural log (nats); dividing by ln 2 ≈ 0.693 converts to bits.
- **Per-token loss trap:** a model with a bigger vocabulary has a higher loss per token, but each token covers more text. Never compare raw losses across tokenizers.
- **Epoch:** one full pass over the training data. Same steps with BPE = more passes (~31 vs ~14), because the book is shorter in tokens, which means more memorizing.
- **Generation speed:** BPE writes ~2.2 characters per step, so it writes the same text ~2.4× faster.

## Lesson 16: Modern upgrades

- **Fast attention (`F.scaled_dot_product_attention`):** all heads computed at once as an extra batch dimension. Same math; on GPUs it's much faster and uses far less memory, on our CPU only slightly faster.
- **Weight tying:** the output layer reuses the token embedding table, saving V × C parameters.
- **GPT-2 initialization:** weights start small (spread 0.02); layers that add onto the residual path start smaller still (÷ √(2 × layers)). Gives a starting loss right at ln(V).
- **Learning-rate warm-up:** start with tiny steps for the first ~100 steps, while the model is still random.
- **Cosine decay:** after warm-up, the learning rate glides down a smooth curve to a small minimum, so the last steps are fine-tuning.
- **Weight decay:** gently pulls the big weight grids toward zero each step (not biases or LayerNorm). Discourages memorizing.
- **Gradient norm / clipping:** the overall size of a step's gradient; clipping scales it down to at most 1.0 so a rare odd batch can't throw training off.
- **Keep-awake request:** the training script asks Windows not to sleep while it runs (closing the lid can still force sleep).
- **RMSNorm, RoPE, SwiGLU, grouped-query attention:** what newer models (Llama etc.) use instead of LayerNorm, learned positions, ReLU feed-forward and per-head keys/values.
