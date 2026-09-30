from __future__ import annotations

import json

from audit_results import load_runs


def main() -> None:
    df = load_runs()
    auc = 100.0 * df.groupby(["method", "dataset"])["convergence_auc"].mean().unstack()
    auc["mean"] = 100.0 * df.groupby("method")["convergence_auc"].mean()
    wins = (
        df.groupby(["dataset", "regime", "method"])["convergence_auc"].mean()
        .reset_index()
    )
    wins["best"] = wins.groupby(["dataset", "regime"])["convergence_auc"].transform("max")
    wins = wins.assign(win=lambda x: (x.convergence_auc == x.best)).groupby("method").win.sum()
    payload = {
        method: {
            "cifar10": float(row["cifar10"]),
            "cifar100": float(row["cifar100"]),
            "officehome": float(row["officehome"]),
            "mean": float(row["mean"]),
            "wins": int(wins[method]),
        }
        for method, row in auc.iterrows()
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
