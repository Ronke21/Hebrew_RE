"""
Dataset statistics for the Hebrew RE (Crocodile) project.
Runs a full analysis for each dataset separately:
  dataset_statistics/gold/   — gold 500-row labelled dataset
  dataset_statistics/full/   — full 3.1M-row dataset (complete)
Also saves a combined text-length comparison graph.
"""

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── config ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
GOLD_CSV    = ROOT / "data/crocodile_heb25_gold_500.csv"
FULL_CSV    = ROOT / "data/crocodile_heb25_full_dataset_3124k.csv"
BASE_OUT    = Path(__file__).resolve().parent
GOLD_OUT    = BASE_OUT / "gold"
FULL_OUT    = BASE_OUT / "full"

TOP_N_PRED  = 20

# ── style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
LABEL_FONT = 9
TITLE_FONT = 11

for d in (GOLD_OUT, FULL_OUT):
    d.mkdir(parents=True, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────
def wrap(label: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(label, width))


def save(fig, out_dir: Path, name: str):
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(BASE_OUT.parent)}")


def section(title: str, width: int = 70) -> str:
    bar = "=" * width
    return f"\n{bar}\n{title}\n{bar}\n"


def write_report(lines: list, out_dir: Path):
    path = out_dir / "statistics.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {path.relative_to(BASE_OUT.parent)}")


# ══════════════════════════════════════════════════════════════════════════════
# GOLD DATASET
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== GOLD DATASET ===")
gold = pd.read_csv(GOLD_CSV)
gold["text_chars"]    = gold["text"].str.len()
gold["text_words"]    = gold["text"].str.split().str.len()
gold["subject_chars"] = gold["subject"].str.len()
gold["object_chars"]  = gold["object"].str.len()

rp = gold["relation_present"].value_counts().sort_index()

# ── text report ────────────────────────────────────────────────────────────────
L = []
L.append("Hebrew RE — Gold Dataset Statistics")
L.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
L.append(f"Source: {GOLD_CSV.name}")

L.append(section("1. Overview"))
L.append(f"  Rows    : {len(gold):,}")
L.append(f"  Columns : {list(gold.columns)}")
L.append(f"  Unique predicates : {gold['predicate'].nunique()}")

L.append(section("2. relation_present Distribution"))
for val, cnt in rp.items():
    label = "present (1)" if val == 1 else "absent  (0)"
    L.append(f"  {label}  :  {cnt:>4}  ({cnt/len(gold)*100:.1f}%)")

L.append(section("3. Text Length"))
for desc, col in [("Characters", "text_chars"), ("Words", "text_words")]:
    s = gold[col].describe()
    L.append(f"\n  {desc}")
    L.append(f"    mean={s['mean']:.1f}  median={s['50%']:.1f}  std={s['std']:.1f}"
             f"  min={s['min']:.0f}  max={s['max']:.0f}")
L.append("\n  Text chars by relation_present:")
for val, grp in gold.groupby("relation_present"):
    label = "present (1)" if val == 1 else "absent  (0)"
    s = grp["text_chars"].describe()
    L.append(f"    {label}  mean={s['mean']:.1f}  median={s['50%']:.1f}  std={s['std']:.1f}")

L.append(section(f"4. Top {TOP_N_PRED} Predicates"))
gold_pred = gold["predicate"].value_counts().head(TOP_N_PRED)
L.append(f"  {'Predicate':<38} {'Count':>6}  {'%':>6}")
L.append("  " + "-" * 55)
for pred, cnt in gold_pred.items():
    L.append(f"  {pred:<38} {cnt:>6}  ({cnt/len(gold)*100:.1f}%)")

L.append(section(f"5. relation_present Rate by Predicate  (top {TOP_N_PRED})"))
top_pred = gold["predicate"].value_counts().head(TOP_N_PRED).index
grp_stat = (
    gold[gold["predicate"].isin(top_pred)]
    .groupby("predicate")["relation_present"]
    .agg(total="count", present="sum")
)
grp_stat["pct"] = grp_stat["present"] / grp_stat["total"] * 100
grp_stat = grp_stat.sort_values("pct", ascending=False)
L.append(f"  {'Predicate':<38} {'total':>6} {'present':>8} {'%present':>9}")
L.append("  " + "-" * 65)
for pred, row in grp_stat.iterrows():
    L.append(f"  {pred:<38} {int(row.total):>6} {int(row.present):>8} {row.pct:>8.1f}%")

L.append(section("6. Subject & Object Character Lengths"))
for col in ["subject_chars", "object_chars"]:
    s = gold[col].describe()
    L.append(f"\n  {col}")
    L.append(f"    mean={s['mean']:.1f}  median={s['50%']:.1f}  std={s['std']:.1f}"
             f"  min={s['min']:.0f}  max={s['max']:.0f}")

write_report(L, GOLD_OUT)

# ── graphs ─────────────────────────────────────────────────────────────────────
# G1 — relation_present bar
fig, ax = plt.subplots(figsize=(5, 4))
colors = ["#d62728", "#2ca02c"]
counts = [rp.get(0, 0), rp.get(1, 0)]
bars = ax.bar(["Absent (0)", "Present (1)"], counts, color=colors, width=0.5, edgecolor="white")
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{cnt}\n({cnt/len(gold)*100:.1f}%)", ha="center", va="bottom", fontsize=LABEL_FONT)
ax.set_title("relation_present Distribution  (Gold, n=500)", fontsize=TITLE_FONT)
ax.set_ylabel("Count")
ax.set_ylim(0, max(counts) * 1.28)
save(fig, GOLD_OUT, "01_relation_present_bar.png")

# G2 — relation_present pie
fig, ax = plt.subplots(figsize=(5, 4))
ax.pie(counts, labels=["Absent (0)", "Present (1)"], colors=colors,
       autopct="%1.1f%%", startangle=140,
       wedgeprops=dict(edgecolor="white", linewidth=1.5))
ax.set_title("relation_present Distribution  (Gold, n=500)", fontsize=TITLE_FONT)
save(fig, GOLD_OUT, "02_relation_present_pie.png")

# G3 — text char histogram
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(gold["text_chars"], bins=30, color="#1f77b4", edgecolor="white")
ax.axvline(gold["text_chars"].mean(), color="red", linestyle="--", label=f"mean={gold['text_chars'].mean():.0f}")
ax.axvline(gold["text_chars"].median(), color="orange", linestyle="--", label=f"median={gold['text_chars'].median():.0f}")
ax.set_xlabel("Text length (characters)")
ax.set_ylabel("Count")
ax.set_title("Text Character Length — Gold Dataset", fontsize=TITLE_FONT)
ax.legend()
save(fig, GOLD_OUT, "03_text_char_length.png")

# G4 — text word histogram
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(gold["text_words"], bins=30, color="#1f77b4", edgecolor="white")
ax.axvline(gold["text_words"].mean(), color="red", linestyle="--", label=f"mean={gold['text_words'].mean():.0f}")
ax.axvline(gold["text_words"].median(), color="orange", linestyle="--", label=f"median={gold['text_words'].median():.0f}")
ax.set_xlabel("Word count")
ax.set_ylabel("Count")
ax.set_title("Text Word Count — Gold Dataset", fontsize=TITLE_FONT)
ax.legend()
save(fig, GOLD_OUT, "04_text_word_count.png")

# G5 — text length by relation_present (box)
fig, ax = plt.subplots(figsize=(5, 4))
bp = ax.boxplot(
    [gold.loc[gold["relation_present"] == 0, "text_chars"],
     gold.loc[gold["relation_present"] == 1, "text_chars"]],
    tick_labels=["Absent (0)", "Present (1)"],
    patch_artist=True, medianprops=dict(color="black", linewidth=2)
)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel("Text length (characters)")
ax.set_title("Text Length by relation_present  (Gold)", fontsize=TITLE_FONT)
save(fig, GOLD_OUT, "05_text_length_by_relation_present.png")

# G6 — top-20 predicates bar (gold)
fig, ax = plt.subplots(figsize=(10, 6))
pred_labels = [wrap(p) for p in gold_pred.index]
ax.barh(pred_labels[::-1], gold_pred.values[::-1], color="#1f77b4", edgecolor="white")
ax.set_xlabel("Count")
ax.set_title(f"Top {TOP_N_PRED} Predicates — Gold Dataset (n=500)", fontsize=TITLE_FONT)
ax.tick_params(axis="y", labelsize=LABEL_FONT)
save(fig, GOLD_OUT, "06_predicate_distribution.png")

# G7 — stacked bar: relation_present rate per predicate (top 15)
top15 = gold["predicate"].value_counts().head(15).index.tolist()
ct = (
    gold[gold["predicate"].isin(top15)]
    .groupby(["predicate", "relation_present"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=[0, 1])
)
ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
ct_norm = ct_norm.loc[ct_norm[1].sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(11, 5))
xlabels = [wrap(p, 20) for p in ct_norm.index]
x = np.arange(len(xlabels))
w = 0.6
ax.bar(x, ct_norm[0], w, label="Absent (0)", color="#d62728", alpha=0.8, edgecolor="white")
ax.bar(x, ct_norm[1], w, bottom=ct_norm[0], label="Present (1)", color="#2ca02c", alpha=0.8, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=LABEL_FONT, rotation=30, ha="right")
ax.set_ylabel("Percentage (%)")
ax.set_ylim(0, 118)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_title("relation_present Rate by Predicate  (Gold, top 15)", fontsize=TITLE_FONT)
ax.legend(loc="upper right")
for i, (pred, row) in enumerate(ct_norm.iterrows()):
    ax.text(i, 103, f"n={int(ct.loc[pred].sum())}", ha="center", va="bottom", fontsize=7)
save(fig, GOLD_OUT, "07_predicate_relation_present_stacked.png")

# G8 — heatmap predicate × relation_present
top20 = gold["predicate"].value_counts().head(TOP_N_PRED).index.tolist()
heat = (
    gold[gold["predicate"].isin(top20)]
    .groupby(["predicate", "relation_present"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=[0, 1])
)
heat.index = [wrap(p, 22) for p in heat.index]
fig, ax = plt.subplots(figsize=(6, 9))
sns.heatmap(heat, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
            linecolor="white", ax=ax, cbar_kws={"label": "Count"})
ax.set_xlabel("relation_present")
ax.set_ylabel("")
ax.set_title(f"Predicate × relation_present  (Gold, top {TOP_N_PRED})", fontsize=TITLE_FONT)
ax.tick_params(axis="y", labelsize=LABEL_FONT)
save(fig, GOLD_OUT, "08_predicate_relation_heatmap.png")

# G9 — subject / object length by relation_present
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, col, title in [
    (axes[0], "subject_chars", "Subject"),
    (axes[1], "object_chars",  "Object"),
]:
    bp = ax.boxplot(
        [gold.loc[gold["relation_present"] == 0, col],
         gold.loc[gold["relation_present"] == 1, col]],
        tick_labels=["Absent (0)", "Present (1)"],
        patch_artist=True, medianprops=dict(color="black", linewidth=2)
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title(f"{title} Length by relation_present", fontsize=TITLE_FONT)
    ax.set_ylabel("Characters")
fig.suptitle("Subject & Object Character Length  (Gold)", fontsize=TITLE_FONT + 1)
fig.tight_layout()
save(fig, GOLD_OUT, "09_subject_object_length.png")


# ══════════════════════════════════════════════════════════════════════════════
# FULL DATASET
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== FULL DATASET ===")
print("  Loading full dataset (this may take a minute) …")
full = pd.read_csv(FULL_CSV)
full["text_chars"] = full["text"].str.len()
full["text_words"] = full["text"].str.split().str.len()
full["subject_chars"] = full["subject"].str.len()
full["object_chars"]  = full["object"].str.len()
print(f"  Loaded {len(full):,} rows.")

# ── text report ────────────────────────────────────────────────────────────────
L = []
L.append("Hebrew RE — Full Dataset Statistics")
L.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
L.append(f"Source: {FULL_CSV.name}")

L.append(section("1. Overview"))
L.append(f"  Total rows  : {len(full):,}")
L.append(f"  Columns     : {list(full.columns)}")
L.append(f"  Unique predicates : {full['predicate'].nunique()}")
L.append("  Note: No relation_present labels — distant-supervision only.")

L.append(section("2. Text Length"))
for desc, col in [("Characters", "text_chars"), ("Words", "text_words")]:
    s = full[col].describe()
    L.append(f"\n  {desc}")
    L.append(f"    mean={s['mean']:.1f}  median={s['50%']:.1f}  std={s['std']:.1f}"
             f"  min={s['min']:.0f}  max={s['max']:.0f}")

L.append(section(f"3. Top {TOP_N_PRED} Predicates"))
full_pred = full["predicate"].value_counts().head(TOP_N_PRED)
L.append(f"  {'Predicate':<38} {'Count':>7}  {'%':>6}")
L.append("  " + "-" * 56)
for pred, cnt in full_pred.items():
    L.append(f"  {pred:<38} {cnt:>7}  ({cnt/len(full)*100:.1f}%)")

L.append(section("4. Subject & Object Character Lengths"))
for col in ["subject_chars", "object_chars"]:
    s = full[col].describe()
    L.append(f"\n  {col}")
    L.append(f"    mean={s['mean']:.1f}  median={s['50%']:.1f}  std={s['std']:.1f}"
             f"  min={s['min']:.0f}  max={s['max']:.0f}")

write_report(L, FULL_OUT)

# ── graphs ─────────────────────────────────────────────────────────────────────
# G1 — text char histogram
fig, ax = plt.subplots(figsize=(8, 4))
clipped = full["text_chars"].clip(upper=5000)
ax.hist(clipped, bins=50, color="#ff7f0e", edgecolor="white")
ax.axvline(full["text_chars"].mean(), color="red", linestyle="--",
           label=f"mean={full['text_chars'].mean():.0f}")
ax.axvline(full["text_chars"].median(), color="blue", linestyle="--",
           label=f"median={full['text_chars'].median():.0f}")
ax.set_xlabel("Text length (characters, clipped at 5000)")
ax.set_ylabel("Count")
ax.set_title(f"Text Character Length — Full Dataset (n={len(full):,})", fontsize=TITLE_FONT)
ax.legend()
save(fig, FULL_OUT, "01_text_char_length.png")

# G2 — text word histogram
fig, ax = plt.subplots(figsize=(8, 4))
clipped_w = full["text_words"].clip(upper=800)
ax.hist(clipped_w, bins=50, color="#ff7f0e", edgecolor="white")
ax.axvline(full["text_words"].mean(), color="red", linestyle="--",
           label=f"mean={full['text_words'].mean():.0f}")
ax.axvline(full["text_words"].median(), color="blue", linestyle="--",
           label=f"median={full['text_words'].median():.0f}")
ax.set_xlabel("Word count (clipped at 800)")
ax.set_ylabel("Count")
ax.set_title(f"Text Word Count — Full Dataset (n={len(full):,})", fontsize=TITLE_FONT)
ax.legend()
save(fig, FULL_OUT, "02_text_word_count.png")

# G3 — top-20 predicates bar
fig, ax = plt.subplots(figsize=(10, 6))
pred_labels = [wrap(p) for p in full_pred.index]
ax.barh(pred_labels[::-1], full_pred.values[::-1], color="#ff7f0e", edgecolor="white")
ax.set_xlabel("Count")
ax.set_title(f"Top {TOP_N_PRED} Predicates — Full Dataset (n={len(full):,})", fontsize=TITLE_FONT)
ax.tick_params(axis="y", labelsize=LABEL_FONT)
save(fig, FULL_OUT, "03_predicate_distribution.png")

# G4 — predicate distribution as pie (top 15 + other)
top15_full = full["predicate"].value_counts().head(15)
other_cnt = full["predicate"].value_counts().iloc[15:].sum()
pie_vals = list(top15_full.values) + [other_cnt]
pie_lbls = [wrap(p, 15) for p in top15_full.index] + ["Other"]
fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(
    pie_vals, labels=pie_lbls, autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
    startangle=140, wedgeprops=dict(edgecolor="white", linewidth=1)
)
for t in texts:
    t.set_fontsize(8)
ax.set_title(f"Predicate Distribution — Full Dataset (n={len(full):,})", fontsize=TITLE_FONT)
save(fig, FULL_OUT, "04_predicate_pie.png")

# G5 — subject / object length box
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, col, title in [
    (axes[0], "subject_chars", "Subject"),
    (axes[1], "object_chars",  "Object"),
]:
    bp = ax.boxplot(
        [full[col].dropna()],
        tick_labels=["All"],
        patch_artist=True, medianprops=dict(color="black", linewidth=2)
    )
    bp["boxes"][0].set_facecolor("#ff7f0e")
    bp["boxes"][0].set_alpha(0.6)
    ax.set_title(f"{title} Character Length", fontsize=TITLE_FONT)
    ax.set_ylabel("Characters")
fig.suptitle(f"Subject & Object Length — Full Dataset (n={len(full):,})", fontsize=TITLE_FONT + 1)
fig.tight_layout()
save(fig, FULL_OUT, "05_subject_object_length.png")


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED — text length comparison
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== COMBINED COMPARISON ===")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# characters
ax = axes[0]
bins_c = np.linspace(0, 4000, 55)
ax.hist(gold["text_chars"].clip(upper=4000), bins=bins_c, alpha=0.65,
        label=f"Gold (n={len(gold):,})", color="#1f77b4", edgecolor="white")
ax.hist(full["text_chars"].clip(upper=4000), bins=bins_c, alpha=0.45,
        label=f"Full (n={len(full):,})", color="#ff7f0e", edgecolor="white")
ax.set_xlabel("Characters (clipped at 4000)")
ax.set_ylabel("Count")
ax.set_title("Text Character Length", fontsize=TITLE_FONT)
ax.legend()

# words
ax = axes[1]
bins_w = np.linspace(0, 600, 55)
ax.hist(gold["text_words"].clip(upper=600), bins=bins_w, alpha=0.65,
        label=f"Gold (n={len(gold):,})", color="#1f77b4", edgecolor="white")
ax.hist(full["text_words"].clip(upper=600), bins=bins_w, alpha=0.45,
        label=f"Full (n={len(full):,})", color="#ff7f0e", edgecolor="white")
ax.set_xlabel("Words (clipped at 600)")
ax.set_ylabel("Count")
ax.set_title("Text Word Count", fontsize=TITLE_FONT)
ax.legend()

fig.suptitle("Gold vs Full Dataset — Text Length Comparison", fontsize=TITLE_FONT + 1)
fig.tight_layout()
save(fig, BASE_OUT, "combined_text_length_comparison.png")

print(f"\nDone.\n  Gold stats  → {GOLD_OUT}/\n  Full stats  → {FULL_OUT}/")
