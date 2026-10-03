"""
Train, tune, calibrate, evaluate and explain the flight-delay model.

    python src/train.py            # full run (40 Optuna trials)
    python src/train.py --trials 5 # quick run

Outputs:
    reports/metrics.json
    reports/figures/*.png
    models/lgbm_delay_model.txt
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, precision_recall_curve, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL, NUMERIC, TARGET, build_features, time_split

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)
FEATURES = CATEGORICAL + NUMERIC
SEED = 42


def evaluate(y, p) -> dict:
    top = np.quantile(p, 0.9)
    return {
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "log_loss": round(log_loss(y, p), 4),
        "precision_top10pct": round(float(y[p >= top].mean()), 4),
        "base_rate": round(float(y.mean()), 4),
    }


def baseline(train, valid, test):
    """Logistic regression with one-hot categoricals and scaled numerics."""
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                              StandardScaler()), NUMERIC),
    ])
    model = make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.5))
    fit = pd.concat([train, valid])
    model.fit(fit[FEATURES].astype({c: str for c in CATEGORICAL}), fit[TARGET])
    return model.predict_proba(test[FEATURES].astype({c: str for c in CATEGORICAL}))[:, 1]


def tune(train, valid, n_trials):
    dtrain = lgb.Dataset(train[FEATURES], train[TARGET], categorical_feature=CATEGORICAL,
                         params={"feature_pre_filter": False})
    dvalid = lgb.Dataset(valid[FEATURES], valid[TARGET], reference=dtrain)

    def objective(trial):
        params = {
            "objective": "binary", "metric": "auc", "verbosity": -1, "seed": SEED,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
            "cat_smooth": trial.suggest_float("cat_smooth", 1, 50),
        }
        booster = lgb.train(params, dtrain, 2000, valid_sets=[dvalid],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        trial.set_user_attr("best_iteration", booster.best_iteration)
        return booster.best_score["valid_0"]["auc"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study


def main(n_trials):
    df = build_features()
    train, valid, test = time_split(df)
    print(f"train {len(train):,} | valid {len(valid):,} | test {len(test):,}")

    # 1. Baseline
    p_lr = baseline(train, valid, test)

    # 2. Tuned LightGBM (tuned on Sep-Oct, never sees Nov-Dec)
    study = tune(train, valid, n_trials)
    best = {**study.best_params, "objective": "binary", "verbosity": -1, "seed": SEED, "bagging_freq": 1}
    n_iter = study.best_trial.user_attrs["best_iteration"]
    booster = lgb.train(best, lgb.Dataset(train[FEATURES], train[TARGET],
                                          categorical_feature=CATEGORICAL), n_iter)
    booster.save_model(str(ROOT / "models" / "lgbm_delay_model.txt"))

    # 3. Isotonic calibration fitted on the validation months
    iso = IsotonicRegression(out_of_bounds="clip").fit(booster.predict(valid[FEATURES]), valid[TARGET])
    p_raw = booster.predict(test[FEATURES])
    p_gbm = iso.predict(p_raw)

    y = test[TARGET].to_numpy()
    metrics = {
        "split": {"train": "Jan-Aug", "valid": "Sep-Oct", "test": "Nov-Dec",
                  "rows": {"train": len(train), "valid": len(valid), "test": len(test)}},
        "logistic_regression": evaluate(y, p_lr),
        "lightgbm_uncalibrated": evaluate(y, p_raw),
        "lightgbm_calibrated": evaluate(y, p_gbm),
        "optuna": {"trials": n_trials, "best_valid_auc": round(study.best_value, 4),
                   "best_params": study.best_params, "best_iteration": n_iter},
    }

    # 4. Ablation: how much do the aircraft-rotation features add?
    rot = ["prev_leg_dep_delay", "has_prev_leg", "mins_since_prev_leg", "leg_of_day"]
    feats_no_rot = [c for c in FEATURES if c not in rot]
    b2 = lgb.train(best, lgb.Dataset(train[feats_no_rot], train[TARGET],
                                     categorical_feature=CATEGORICAL), n_iter)
    metrics["ablation_without_rotation_features"] = evaluate(y, b2.predict(test[feats_no_rot]))

    # 5. Slice: test AUC by month (December has holiday-season shift)
    metrics["test_auc_by_month"] = {
        int(m): round(roc_auc_score(g[TARGET], iso.predict(booster.predict(g[FEATURES]))), 4)
        for m, g in test.groupby("month")
    }

    # ---------- figures ----------
    # SHAP on a sample of test flights
    sample = test.sample(5000, random_state=SEED)
    sv = shap.TreeExplainer(booster).shap_values(sample[FEATURES])
    sv = sv[1] if isinstance(sv, list) else sv
    plt.figure()
    shap.summary_plot(sv, sample[FEATURES], max_display=15, show=False)
    plt.title("What drives predicted delay risk (SHAP, test set)", loc="left", fontweight="bold")
    plt.tight_layout(); plt.savefig(FIG / "shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()

    imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    metrics["top_features_mean_abs_shap"] = imp.head(10).round(4).to_dict()

    # Calibration
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in [("Logistic regression", p_lr), ("LightGBM (raw)", p_raw), ("LightGBM (calibrated)", p_gbm)]:
        fx, fy = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(fy, fx, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed delay rate")
    ax.set_title("Calibration on Nov-Dec test flights", loc="left", fontweight="bold")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "calibration.png", dpi=150); plt.close(fig)

    # Precision-recall
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in [("Logistic regression", p_lr), ("LightGBM (calibrated)", p_gbm)]:
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=f"{name} (AP={average_precision_score(y, p):.3f})")
    ax.axhline(y.mean(), ls="--", color="grey", label=f"Base rate ({y.mean():.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-recall, Nov-Dec test flights", loc="left", fontweight="bold")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "precision_recall.png", dpi=150); plt.close(fig)

    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "optuna"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    main(ap.parse_args().trials)
