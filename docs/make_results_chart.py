"""Draw docs/results.png: the whole project's progress on one scale (numbers from notes/progress.md).

Run:  uv run python docs/make_results_chart.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = [  # (model, bits per character on the same held-out text)
    ("Knows nothing", 6.41),
    ("Character frequencies", 4.39),
    ("Bigram", 3.47),
    ("1 attention head", 3.33),
    ("4 heads + feed-forward", 2.74),
    ("First GPT (4 blocks)", 2.18),
    ("Character-level GPT", 1.84),
    ("BPE tokenizer GPT", 1.69),
    ("BPE GPT + modern training", 1.62),
]

fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
fig.patch.set_facecolor("#0b0f17")
ax.set_facecolor("#0b0f17")
names = [n for n, _ in RESULTS][::-1]
values = [v for _, v in RESULTS][::-1]
bars = ax.barh(names, values, color=["#b28dff"] + ["#7cc4ff"] * (len(values) - 1), height=0.62)  # final model highlighted
for bar, v in zip(bars, values):
    ax.text(v + 0.08, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", color="#e6eaf2", fontsize=10)
ax.set_xlim(0, 7.2)
ax.set_xlabel("bits per character on held-out Tesla text (lower is better)", color="#93a0b8")
ax.set_title("TinyGPT Tesla: from random guessing to a trained GPT, built step by step",
             color="#e6eaf2", fontsize=13, loc="left", pad=12)
ax.tick_params(colors="#e6eaf2", labelsize=10)
ax.tick_params(axis="x", colors="#93a0b8")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#26314a")
fig.text(0.01, 0.01, "All models trained on a laptop CPU. Final model: 874K parameters, 32 minutes of training.",
         color="#93a0b8", fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = Path(__file__).parent / "results.png"
fig.savefig(out, facecolor=fig.get_facecolor())
print("saved", out)
