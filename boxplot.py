import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def main():
    fnames: list[str] = [f for f in os.listdir(".") if ".csv" in f]
    for fname in fnames:
        df = pd.read_csv(fname)
        df['alg'] = df['alg'].str.upper()

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=df, x="alg", y="fitness", ax=ax)
        ax.set_xlabel("Algoritmo", fontsize=12)
        ax.set_ylabel("Fitness", fontsize=12)
        ax.set_title(fname.replace(".csv", "").replace("_", " "), fontsize=13)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()

        foutput = fname.replace(".csv", ".png")
        fig.savefig(foutput, dpi=150)
        plt.close(fig)

if __name__ == '__main__':
    main()

