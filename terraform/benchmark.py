import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path.home() / "ml-benchmark" / "creditcard.csv"
RESULT_PATH = Path.home() / "ml-benchmark" / "benchmark_result.json"


def now():
    return time.perf_counter()


def main():
    if not DATA_PATH.exists():
        raise SystemExit(
            f"Dataset not found: {DATA_PATH}\n"
            "Download it first with: kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/"
        )

    started = now()
    df = pd.read_csv(DATA_PATH)
    load_time = now() - started

    x = df.drop(columns=["Class"])
    y = df["Class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    train_started = now()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )
    training_time = now() - train_started

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    one_row = x_test.iloc[:1]
    latency_started = now()
    model.predict_proba(one_row)
    one_row_latency_ms = (now() - latency_started) * 1000

    batch = x_test.iloc[:1000]
    throughput_started = now()
    model.predict_proba(batch)
    batch_time = now() - throughput_started

    results = {
        "rows": int(len(df)),
        "features": int(x.shape[1]),
        "load_data_seconds": round(load_time, 4),
        "training_seconds": round(training_time, 4),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "auc_roc": round(float(roc_auc_score(y_test, probabilities)), 6),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "f1_score": round(float(f1_score(y_test, predictions)), 6),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, predictions)), 6),
        "inference_latency_1_row_ms": round(one_row_latency_ms, 4),
        "inference_throughput_1000_rows_per_second": round(1000 / batch_time, 2),
    }

    RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== LightGBM CPU Benchmark Results ===")
    for key, value in results.items():
        print(f"{key}: {value}")
    print(f"\nSaved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
