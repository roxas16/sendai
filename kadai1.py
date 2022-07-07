import argparse
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", type=str, help="plese set output dir")

    parser.add_argument("--in_file", default="report.csv", type=str, help="plese set filename")
    parser.add_argument("--out_file", default="out.png", type=str, help="plese set filename")

    return parser.parse_args()


def get_coeffs(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray]:
    a = np.cov(x, y)[0, 1] / x.var()
    b = y.mean() - a * x.mean()

    return a, b


def main():
    args = make_parser()
    parent = Path(args.out_dir)
    parent.mkdir(parents=True, exist_ok=True)

    points = pd.read_csv(args.in_file, header=None, names=["x", "y"])
    a, b = get_coeffs(x=points["x"], y=points["y"])

    x = np.arange(-110, 111)
    y = a * x + b

    sns.lineplot(x, y, color="orange")
    sns.scatterplot(data=points, x="x", y="y")
    plt.savefig(parent / args.out_file)


if __name__ == "__main__":
    main()
