import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

GREEDY_ONLY = {"GREEDY"}

def main():
    fnames: list[str] = sorted([f for f in os.listdir(".") if f.endswith(".csv")])
    for fname in fnames:
        df = pd.read_csv(fname)
        df['alg'] = df['alg'].str.upper()
        title = fname.replace(".csv", "").replace("_", " ")

        # ── Figure 1: all algorithms including Greedy ─────────────────────────
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=df, x="alg", y="fitness", ax=ax)
        ax.set_xlabel("Algoritmo", fontsize=12)
        ax.set_ylabel("Fitness", fontsize=12)
        ax.set_title(f"{title} — todos los algoritmos", fontsize=13)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        fig.savefig(fname.replace(".csv", "_all.png"), dpi=150)
        plt.close(fig)

        # ── Figure 2: without Greedy ──────────────────────────────────────────
        df_no_greedy = df[~df['alg'].isin(GREEDY_ONLY)]
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=df_no_greedy, x="alg", y="fitness", ax=ax)
        ax.set_xlabel("Algoritmo", fontsize=12)
        ax.set_ylabel("Fitness", fontsize=12)
        ax.set_title(f"{title} — sin Greedy", fontsize=13)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        fig.savefig(fname.replace(".csv", "_no_greedy.png"), dpi=150)
        plt.close(fig)

        print(f"Generated: {fname.replace('.csv', '_all.png')} and {fname.replace('.csv', '_no_greedy.png')}")

if __name__ == '__main__':
    main()
