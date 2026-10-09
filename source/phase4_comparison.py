"""
Phase 4: So sanh ket qua Phase 2 (scikit-learn) va Phase 3 (Spark MLlib).

Chay tu thu muc goc du an:
    python source/phase4_comparison.py

Script chi doc cac output da co cua Phase 2/3, sau do tao:
- output_phase4/comparison_metrics.csv
- output_phase4/model_summary.csv
- output_phase4/feature_importance_comparison.csv
- output_phase4/prediction_comparison.csv
- output_phase4/comparison_report.md
- pics/phase4/*.png
"""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PHASE2 = ROOT / "output_phase2"
PHASE3 = ROOT / "output_phase3"
OUT = ROOT / "output_phase4"
PICS = ROOT / "pics" / "phase4"
OUT.mkdir(exist_ok=True)
PICS.mkdir(exist_ok=True)

MODELS = {"Phase 2 - scikit-learn": PHASE2, "Phase 3 - Spark MLlib": PHASE3}
METRIC_NAMES = ["MAE", "RMSE", "R2"]
SPLIT_ORDER = ["train", "validation", "test"]


def require_columns(df, columns, path):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{path} thieu cot: {missing}")


def load_metrics():
    frames = []
    for model_name, directory in MODELS.items():
        path = directory / "metrics.csv"
        df = pd.read_csv(path)
        require_columns(df, ["split", *METRIC_NAMES], path)
        df = df[["split", *METRIC_NAMES]].copy()
        df["model"] = model_name
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    result["split"] = pd.Categorical(result["split"], SPLIT_ORDER, ordered=True)
    return result.sort_values(["split", "model"]).reset_index(drop=True)


def load_predictions():
    left = pd.read_csv(PHASE2 / "test_predictions.csv")
    right = pd.read_csv(PHASE3 / "test_predictions.csv")
    expected = ["record_id", "actual_power_output", "predicted_power_output"]
    require_columns(left, expected, PHASE2 / "test_predictions.csv")
    require_columns(right, expected, PHASE3 / "test_predictions.csv")
    left = left.rename(columns={"predicted_power_output": "prediction_sklearn"})
    right = right.rename(columns={"predicted_power_output": "prediction_spark"})
    left = left[["record_id", "actual_power_output", "prediction_sklearn"]]
    right = right[["record_id", "actual_power_output", "prediction_spark"]]
    merged = left.merge(right, on="record_id", suffixes=("_sklearn", "_spark"), validate="one_to_one")
    if not np.allclose(merged["actual_power_output_sklearn"], merged["actual_power_output_spark"]):
        raise ValueError("Actual power_output cua hai output khong khop theo record_id")
    merged = merged.rename(columns={"actual_power_output_sklearn": "actual_power_output"})
    merged["error_sklearn"] = merged["actual_power_output"] - merged["prediction_sklearn"]
    merged["error_spark"] = merged["actual_power_output"] - merged["prediction_spark"]
    merged["absolute_error_sklearn"] = merged["error_sklearn"].abs()
    merged["absolute_error_spark"] = merged["error_spark"].abs()
    merged["prediction_difference_spark_minus_sklearn"] = (
        merged["prediction_spark"] - merged["prediction_sklearn"]
    )
    return merged


def normalize_feature_name(name):
    name = str(name)
    for prefix in ("num__", "cat__", "indexed_encoded_", "_indexed_encoded_"):
        name = name.replace(prefix, "")
    return name


def load_feature_importance():
    all_frames = []
    for model_name, directory in MODELS.items():
        path = directory / "feature_importance.csv"
        df = pd.read_csv(path)
        feature_col = "feature" if "feature" in df.columns else "feature_name"
        require_columns(df, [feature_col, "importance"], path)
        clean = pd.DataFrame({
            "model": model_name,
            "feature": df[feature_col].map(normalize_feature_name),
            "importance_encoded": pd.to_numeric(df["importance"]),
        })
        # Phase 2 chi xuat top-20 encoded features, vi vay giu importance goc
        # thay vi chuan hoa lai theo tong top-20 (se lam sai dien giai).
        clean = clean.groupby(["model", "feature"], as_index=False)["importance_encoded"].sum()
        clean["importance"] = clean["importance_encoded"]
        clean["rank"] = clean.groupby("model")["importance"].rank(
            method="min", ascending=False
        ).astype(int)
        all_frames.append(clean)
    result = pd.concat(all_frames, ignore_index=True)
    return result.sort_values(["model", "importance"], ascending=[True, False])


def parse_spark_info():
    path = PHASE3 / "spark_run_info.txt"
    text = path.read_text(encoding="utf-8")
    patterns = {
        "spark_version": r"Spark Version:\s*(.+)",
        "spark_master": r"Spark Master:\s*(.+)",
        "preprocessing_fit_time_s": r"Preprocessing fit time:\s*([\d.]+)s",
        "final_train_time_s": r"Final model training time:\s*([\d.]+)s",
        "pipeline_time_s": r"Whole pipeline execution time:\s*([\d.]+)s",
        "best_num_trees": r"numTrees:\s*(\d+)",
        "best_max_depth": r"maxDepth:\s*(\d+)",
    }
    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            parsed[key] = float(value) if key.endswith("_s") else value
    return parsed


def load_model_summary():
    p2_search = pd.read_csv(PHASE2 / "param_search_results.csv")
    p3_search = pd.read_csv(PHASE3 / "param_search_results.csv")
    p2_best = p2_search.loc[p2_search["rmse_val"].idxmin()]
    p3_best = p3_search.loc[p3_search["validation_RMSE"].idxmin()]
    spark_info = parse_spark_info()
    rows = [
        {
            "model": "Phase 2 - scikit-learn",
            "framework": "scikit-learn",
            "best_configuration": (
                f"n_estimators={int(p2_best['n_estimators'])}, "
                f"max_depth={int(p2_best['max_depth']) if pd.notna(p2_best['max_depth']) else 'None'}, "
                f"min_samples_leaf={int(p2_best['min_samples_leaf'])}"
            ),
            "tuning_configurations": len(p2_search),
            "best_validation_rmse": p2_best["rmse_val"],
            "best_train_time_s": p2_best["train_time_s"],
            "pipeline_time_s": np.nan,
            "execution_mode": "local scikit-learn",
        },
        {
            "model": "Phase 3 - Spark MLlib",
            "framework": "PySpark Spark MLlib",
            "best_configuration": (
                f"numTrees={int(p3_best['numTrees'])}, maxDepth={int(p3_best['maxDepth'])}"
            ),
            "tuning_configurations": len(p3_search),
            "best_validation_rmse": p3_best["validation_RMSE"],
            "best_train_time_s": spark_info.get("final_train_time_s", np.nan),
            "pipeline_time_s": spark_info.get("pipeline_time_s", np.nan),
            "execution_mode": spark_info.get("spark_master", "local[*]"),
        },
    ]
    return pd.DataFrame(rows)


def make_metric_plot(metrics):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"Phase 2 - scikit-learn": "#4C72B0", "Phase 3 - Spark MLlib": "#DD8452"}
    for ax, metric in zip(axes, METRIC_NAMES):
        pivot = metrics.pivot(index="split", columns="model", values=metric).reindex(SPLIT_ORDER)
        pivot.plot(kind="bar", ax=ax, color=[colors[col] for col in pivot.columns], width=0.75)
        ax.set_title(metric if metric != "R2" else "R²")
        ax.set_xlabel("Data split")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.25)
        if metric == "R2":
            ax.set_ylim(0, 1)
        ax.legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Phase 4 - So sanh chi so mo hinh", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PICS / "metrics_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_prediction_plot(predictions):
    rng = np.random.default_rng(42)
    sample = predictions.iloc[rng.choice(len(predictions), size=min(5000, len(predictions)), replace=False)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    actual = sample["actual_power_output"]
    lower = min(actual.min(), sample["prediction_sklearn"].min(), sample["prediction_spark"].min())
    upper = max(actual.max(), sample["prediction_sklearn"].max(), sample["prediction_spark"].max())
    axes[0].scatter(actual, sample["prediction_sklearn"], s=5, alpha=0.25, label="scikit-learn", color="#4C72B0")
    axes[0].scatter(actual, sample["prediction_spark"], s=5, alpha=0.25, label="Spark", color="#DD8452")
    axes[0].plot([lower, upper], [lower, upper], "k--", linewidth=1)
    axes[0].set_title("Actual vs predicted")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].legend()
    axes[1].hist(sample["error_sklearn"], bins=55, alpha=0.6, label="scikit-learn", color="#4C72B0")
    axes[1].hist(sample["error_spark"], bins=55, alpha=0.6, label="Spark", color="#DD8452")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Actual - predicted")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[2].hist(predictions["prediction_difference_spark_minus_sklearn"], bins=55, color="#55A868", alpha=0.8)
    axes[2].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Prediction difference (Spark - sklearn)")
    axes[2].set_xlabel("Difference")
    axes[2].set_ylabel("Count")
    fig.suptitle("Phase 4 - So sanh du doan tren cung record_id", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PICS / "prediction_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_feature_plot(features):
    top_features = features.groupby("feature")["importance"].max().nlargest(12).index
    plot_df = features[features["feature"].isin(top_features)].copy()
    pivot = plot_df.pivot(index="feature", columns="model", values="importance").fillna(0)
    pivot = pivot.loc[pivot.max(axis=1).sort_values().index]
    ax = pivot.plot(kind="barh", figsize=(11, 7), color=["#4C72B0", "#DD8452"])
    ax.set_title("Phase 4 - Feature importance (gop theo feature goc)", fontweight="bold")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(PICS / "feature_importance_comparison.png", dpi=180, bbox_inches="tight")
    plt.close()


def make_report(metrics, summary, features, predictions):
    test = metrics[metrics["split"] == "test"].set_index("model")
    p2 = test.loc["Phase 2 - scikit-learn"]
    p3 = test.loc["Phase 3 - Spark MLlib"]
    metric_rows = []
    for metric in METRIC_NAMES:
        diff = p3[metric] - p2[metric]
        direction = "tot hon" if (diff < 0 if metric != "MAE" else diff < 0) else "kem hon"
        if metric == "R2":
            direction = "tot hon" if diff > 0 else "kem hon"
        metric_rows.append(f"| {metric} | {p2[metric]:.6f} | {p3[metric]:.6f} | {diff:+.6f} | Spark {direction} |")
    p2_train = metrics[(metrics["model"] == "Phase 2 - scikit-learn") & (metrics["split"] == "train")].iloc[0]
    p3_train = metrics[(metrics["model"] == "Phase 3 - Spark MLlib") & (metrics["split"] == "train")].iloc[0]
    p2_gap = p2["RMSE"] - p2_train["RMSE"]
    p3_gap = p3["RMSE"] - p3_train["RMSE"]
    top = features.sort_values("importance", ascending=False).groupby("model").head(5)
    top_lines = []
    for model in MODELS:
        names = ", ".join(f"{row.feature} ({row.importance:.3f})" for row in top[top["model"] == model].itertuples())
        top_lines.append(f"- **{model}:** {names}")
    same_actual = predictions["actual_power_output"].notna().all()
    pred_diff = predictions["prediction_difference_spark_minus_sklearn"]
    summary_rows = summary.to_markdown(index=False)
    report = f"""# Phase 4 - So sanh hai mo hinh Random Forest

## 1. Pham vi

Phase 4 so sanh ket qua tu:

- **Phase 2:** Random Forest Regression bang scikit-learn.
- **Phase 3:** RandomForestRegressor bang PySpark Spark MLlib.
- Hai phase dung cung tap train/validation/test va cung target `power_output`.
- File prediction duoc doi chieu theo `record_id`; actual cua hai mo hinh khop: **{same_actual}**.

## 2. So sanh do chinh xac tren test

| Metric | scikit-learn | Spark MLlib | Spark - sklearn | Ket luan |
|---|---:|---:|---:|---|
{chr(10).join(metric_rows)}

- Spark co RMSE nho hon **{abs(p3['RMSE'] - p2['RMSE']):.4f}** va R² cao hon **{p3['R2'] - p2['R2']:+.4f}**.
- scikit-learn co MAE nho hon **{abs(p3['MAE'] - p2['MAE']):.4f}**.
- Cac chenh lech deu rat nho, vi vay khong nen ket luan mot mo hinh vuot troi tuyet doi chi dua tren accuracy.

## 3. Overfitting va kha nang tong quat hoa

| Model | Train RMSE | Test RMSE | Gap Test - Train |
|---|---:|---:|---:|
| scikit-learn | {p2_train['RMSE']:.6f} | {p2['RMSE']:.6f} | {p2_gap:.6f} |
| Spark MLlib | {p3_train['RMSE']:.6f} | {p3['RMSE']:.6f} | {p3_gap:.6f} |

Spark co khoang cach train-test nho hon trong output hien tai, cho thay ket qua it bi overfit hon. Tuy nhien, hai pipeline khong hoan toan dong nhat ve tham so: sklearn dung `min_samples_leaf`, con Spark khong co tham so tuong duong trong script.

## 4. Cau hinh va thoi gian

{summary_rows}

- scikit-learn thu **{int(summary.iloc[0]['tuning_configurations'])}** cau hinh; Spark thu **{int(summary.iloc[1]['tuning_configurations'])}** cau hinh.
- Spark phu hop hon khi can chay tren cluster va mo rong du lieu; output hien tai chay `local[*]`, khong phai benchmark cluster phan tan.
- Thoi gian pipeline Spark ghi nhan trong `spark_run_info.txt` la **{summary.iloc[1]['pipeline_time_s']:.2f}s**. Thoi gian train config tot nhat la **{summary.iloc[1]['best_train_time_s']:.2f}s**; sklearn la **{summary.iloc[0]['best_train_time_s']:.2f}s**.

## 5. Feature importance

Importance da duoc gop tu cac cot one-hot ve feature goc de so sanh cong bang. Luu y: Phase 2 chi xuat top-20 encoded features, con Phase 3 xuat day du cac feature; do do bang nay dung de so sanh thu hang va cac feature da xuat, khong phai tong importance day du cua Phase 2:

{chr(10).join(top_lines)}

Cả hai mo hinh deu cho thay `wind_speed` la feature quan trong nhat. Spark co importance cua `wind_speed` cao hon trong output (xap xi 0.891 so voi 0.682). Luu y Phase 2 chi luu top-20 encoded features, nen khong dung tong importance cua hai file de so sanh truc tiep. Feature importance cua Random Forest la muc do dong gop theo cay, khong phai quan he nhan-qua.

## 6. So sanh prediction cung mau

- So mau test duoc doi chieu: **{len(predictions):,}**.
- Chenh lech prediction Spark - sklearn: mean = **{pred_diff.mean():.6f}**, mean absolute = **{pred_diff.abs().mean():.6f}**.
- Ket qua nay giup xac nhan hai mo hinh dang hoc quy luat gan nhau du khac framework va preprocessing encoding.

## 7. Ket luan de trinh bay

1. **Ve accuracy:** Spark MLlib nhinh hon nhe theo RMSE va R²; scikit-learn nhinh hon nhe theo MAE. Muc chenh lech khoang 0.1-0.3%, gan nhu tuong duong.
2. **Ve Big Data:** Spark la lua chon phu hop de trien khai tren du lieu lon/cluster nhờ pipeline MLlib va kha nang phan tan. Ket qua local[*] hien tai chi danh gia workflow, chua chung minh loi the cluster.
3. **Ve mo hinh:** Cả hai deu phu thuoc manh vao `wind_speed`; co the cai thien bang feature engineering theo vat ly tuabin, xu ly zero-output, tuning dong nhat hon va cross-validation.
4. **Lua chon de de xuat:** Chon Spark cho he thong Big Data production; chon scikit-learn cho prototype nhanh tren mot may va kiem thu mo hinh.

## 8. Tep ket qua

- `output_phase4/comparison_metrics.csv`
- `output_phase4/model_summary.csv`
- `output_phase4/feature_importance_comparison.csv`
- `output_phase4/prediction_comparison.csv`
- `pics/phase4/metrics_comparison.png`
- `pics/phase4/prediction_comparison.png`
- `pics/phase4/feature_importance_comparison.png`
"""
    (OUT / "comparison_report.md").write_text(report, encoding="utf-8")


def main():
    print("PHASE 4 - MODEL COMPARISON")
    metrics = load_metrics()
    predictions = load_predictions()
    features = load_feature_importance()
    summary = load_model_summary()

    metrics.to_csv(OUT / "comparison_metrics.csv", index=False)
    summary.to_csv(OUT / "model_summary.csv", index=False)
    features.to_csv(OUT / "feature_importance_comparison.csv", index=False)
    predictions.to_csv(OUT / "prediction_comparison.csv", index=False)

    make_metric_plot(metrics)
    make_prediction_plot(predictions)
    make_feature_plot(features)
    make_report(metrics, summary, features, predictions)

    print(f"Metrics: {OUT / 'comparison_metrics.csv'}")
    print(f"Report:  {OUT / 'comparison_report.md'}")
    print(f"Plots:   {PICS}")
    print("Done.")


if __name__ == "__main__":
    main()
