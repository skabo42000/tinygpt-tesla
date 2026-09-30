# tinygpt

A small GPT language model built from scratch in PyTorch, one lesson at a time (following Andrej Karpathy's "Let's build GPT").

- `notebooks/`: one notebook per lesson
- `src/tinygpt/`: the finished, clean code
- `notes/glossary.md`: plain-English meanings of every term
- `notes/progress.md`: which lessons are done, and our real loss numbers

## Running things

The Python environment lives **outside OneDrive** at `C:\Users\bosko\.venvs\llm-project`.
VS Code terminals in this folder already know that (see `.vscode/settings.json`).
In any other terminal, run this first:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "C:\Users\bosko\.venvs\llm-project"
```

Then use `uv run ...` as usual.

## Dataset

The model learns from *The Inventions, Researches and Writings of Nikola Tesla* (Project Gutenberg). To download and clean it into `data/input.txt`:

```powershell
uv run python -m tinygpt.download_tesla
```
