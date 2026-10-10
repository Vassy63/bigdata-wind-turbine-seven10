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
import sys

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

# 13 dac trung goc
ORIGINAL_FEATURES = [
    "wind_speed",
    "wind_direction",
    "air_density",
    "temperature",
    "blade_angle",
    "rotor_speed",
    "generator_efficiency",
    "altitude",
    "humidity",
    "region",
    "turbine_id",
    "maintenance_status",
    "technician_note",
]


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


def normalize_feature_name(raw_name):
    """
    Chuyen ten encoded feature ve ten goc trong ORIGINAL_FEATURES.

    Xu ly ca hai kieu dat ten:
      - scikit-learn ColumnTransformer:
          num__<feature>           -> <feature>      vi du: num__wind_speed -> wind_speed
          cat__<feature>_<value>   -> <feature>      vi du: cat__region_Offshore -> region
                                                             cat__turbine_id_TB0011 -> turbine_id
      - Spark MLlib StringIndexer+OneHotEncoder:
          <feature>                                  -> <feature>  (numeric, khong co suffix)
          <feature>_indexed_encoded_<value>          -> <feature>  vi du: region_indexed_encoded_Coastal -> region
          <feature>_indexed_encoded___unknown        -> <feature>
    """
    name = str(raw_name).strip()

    # --- sklearn: co prefix num__ hoac cat__ ---
    sklearn_match = re.match(r'^(?:num__|cat__)(.+)$', name)
    if sklearn_match:
        remainder = sklearn_match.group(1)
        # Kiem tra tung feature goc tu dai den ngan tranh nham
        for feat in sorted(ORIGINAL_FEATURES, key=len, reverse=True):
            if remainder == feat:
                return feat
            if remainder.startswith(feat + "_"):
                return feat
        print(f"[WARN] normalize_feature_name: khong nhan ra sklearn feature '{name}'", file=sys.stderr)
        return name

    # --- Spark: pattern <feature>_indexed_encoded_<value> ---
    spark_match = re.match(r'^(.+)_indexed_encoded_(.+)$', name)
    if spark_match:
        candidate = spark_match.group(1)
        if candidate in ORIGINAL_FEATURES:
            return candidate
        print(
            f"[WARN] normalize_feature_name: Spark feature '{name}' co prefix '{candidate}' khong thuoc ORIGINAL_FEATURES",
            file=sys.stderr,
        )
        return candidate

    # --- Ten goc truc tiep (Spark numeric features khong co suffix) ---
    if name in ORIGINAL_FEATURES:
        return name

    print(f"[WARN] normalize_feature_name: khong nhan ra feature '{name}'", file=sys.stderr)
    return name


def load_feature_importance():
    """
    Doc feature importance tu Phase 2 va Phase 3, cong cac cot one-hot ve 13 dac trung goc.

    Phase 2: output_phase2/feature_importance_full.csv  (80 encoded features)
    Phase 3: output_phase3/feature_importance.csv       (toan bo features da xuat)

    Neu feature_importance_full.csv chua ton tai, raise FileNotFoundError ro rang.
    Khong su dung bang top-20 de thay the va khong chuan hoa lai tong importance.
    """
    p2_full_path = PHASE2 / "feature_importance_full.csv"
    if not p2_full_path.exists():
        raise FileNotFoundError(
            f"Thieu file: {p2_full_path}\n"
            "Phase 4 yeu cau bang feature importance day du (80 dac trung) cua Phase 2. "
            "Khong su dung bang top-20 de thay the. Hay chay lai Phase 2 de xuat file nay."
        )

    source_files = {
        "Phase 2 - scikit-learn": p2_full_path,
        "Phase 3 - Spark MLlib": PHASE3 / "feature_importance.csv",
    }

    all_frames = []
    for model_name, path in source_files.items():
        df = pd.read_csv(path)
        feature_col = "feature" if "feature" in df.columns else "feature_name"
        require_columns(df, [feature_col, "importance"], path)

        raw_features = df[feature_col].tolist()
        importances = pd.to_numeric(df["importance"], errors="coerce").tolist()

        mapped = [normalize_feature_name(f) for f in raw_features]

        clean = pd.DataFrame({
            "model": model_name,
            "feature": mapped,
            "importance_encoded": importances,
        })

        # Loc bo hang __unknown (importance = 0, artifact cua Spark pipeline)
        clean = clean[~clean["feature"].str.endswith("__unknown")]

        # Cong importance theo feature goc
        grouped = (
            clean.groupby(["model", "feature"], as_index=False)["importance_encoded"]
            .sum()
        )
        grouped["importance"] = grouped["importance_encoded"]

        # Kiem tra: phai co dung 13 feature goc
        found_features = set(grouped["feature"].tolist())
        expected_features = set(ORIGINAL_FEATURES)
        missing = expected_features - found_features
        extra = found_features - expected_features
        if missing:
            raise ValueError(
                f"[{model_name}] Thieu {len(missing)} feature goc sau khi gop: {sorted(missing)}\n"
                f"Kiem tra normalize_feature_name() hoac CSV tai: {path}"
            )
        if extra:
            print(
                f"[WARN] [{model_name}] Co {len(extra)} feature la la sau khi gop: {sorted(extra)}\n"
                f"  Cac feature nay se bi loai bo.",
                file=sys.stderr,
            )
            grouped = grouped[grouped["feature"].isin(expected_features)]

        # Kiem tra: importance hop le (huu han, khong am, tong ~ 1)
        imp_vals = grouped["importance"].values
        if not np.all(np.isfinite(imp_vals)):
            raise ValueError(f"[{model_name}] Co gia tri importance khong huu han (NaN/Inf).")
        if np.any(imp_vals < 0):
            raise ValueError(f"[{model_name}] Co gia tri importance am.")
        total = imp_vals.sum()
        if not np.isclose(total, 1.0, atol=0.01):
            print(
                f"[WARN] [{model_name}] Tong importance = {total:.6f} (khong xap xi 1.0).",
                file=sys.stderr,
            )

        grouped["rank"] = grouped["importance"].rank(method="min", ascending=False).astype(int)
        all_frames.append(grouped)

    result = pd.concat(all_frames, ignore_index=True)
    return result.sort_values(["model", "importance"], ascending=[True, False]).reset_index(drop=True)


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
        ax.set_title(metric if metric != "R2" else "R\u00b2")
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
    # Hien thi tat ca 13 feature goc (da gop importance)
    pivot = features.pivot(index="feature", columns="model", values="importance").fillna(0)
    pivot = pivot.loc[pivot.max(axis=1).sort_values().index]
    ax = pivot.plot(kind="barh", figsize=(11, 7), color=["#4C72B0", "#DD8452"])
    ax.set_title("Phase 4 - Feature importance (gop ve 13 feature goc)", fontweight="bold")
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

    mae_diff = p3["MAE"] - p2["MAE"]
    rmse_diff = p3["RMSE"] - p2["RMSE"]
    r2_diff = p3["R2"] - p2["R2"]

    metric_rows = []
    for metric in METRIC_NAMES:
        diff = p3[metric] - p2[metric]
        if metric == "R2":
            direction = "tot hon" if diff > 0 else "kem hon"
        else:
            direction = "tot hon" if diff < 0 else "kem hon"
        metric_rows.append(
            f"| {metric} | {p2[metric]:.6f} | {p3[metric]:.6f} | {diff:+.6f} | Spark {direction} |"
        )

    p2_train = metrics[
        (metrics["model"] == "Phase 2 - scikit-learn") & (metrics["split"] == "train")
    ].iloc[0]
    p3_train = metrics[
        (metrics["model"] == "Phase 3 - Spark MLlib") & (metrics["split"] == "train")
    ].iloc[0]
    p2_gap = p2["RMSE"] - p2_train["RMSE"]
    p3_gap = p3["RMSE"] - p3_train["RMSE"]

    top = features.sort_values("importance", ascending=False).groupby("model").head(5)
    top_lines = []
    for model in MODELS:
        names = ", ".join(
            f"{row.feature} ({row.importance:.4f})"
            for row in top[top["model"] == model].itertuples()
        )
        top_lines.append(f"- **{model}:** {names}")

    imp_totals = features.groupby("model")["importance"].sum()
    p2_total = imp_totals.get("Phase 2 - scikit-learn", float("nan"))
    p3_total = imp_totals.get("Phase 3 - Spark MLlib", float("nan"))

    same_actual = predictions["actual_power_output"].notna().all()
    pred_diff = predictions["prediction_difference_spark_minus_sklearn"]
    summary_rows = summary.to_markdown(index=False)

    # Nhan xet chenh lech cu the (khong dung "0.1-0.3%" chung chung)
    rmse_pct = abs(rmse_diff) / p2["RMSE"] * 100
    mae_pct = abs(mae_diff) / p2["MAE"] * 100
    r2_abs = abs(r2_diff)

    if rmse_diff < 0:
        rmse_comment = (
            f"Spark co RMSE nho hon sklearn {abs(rmse_diff):.4f} "
            f"({rmse_pct:.2f}% tuong doi so voi sklearn)"
        )
    else:
        rmse_comment = (
            f"Spark co RMSE lon hon sklearn {abs(rmse_diff):.4f} "
            f"({rmse_pct:.2f}% tuong doi so voi sklearn)"
        )

    if mae_diff < 0:
        mae_comment = (
            f"Spark co MAE nho hon sklearn {abs(mae_diff):.4f} "
            f"({mae_pct:.2f}% tuong doi)"
        )
    else:
        mae_comment = (
            f"sklearn co MAE nho hon Spark {abs(mae_diff):.4f} "
            f"({mae_pct:.2f}% tuong doi)"
        )

    if r2_diff > 0:
        r2_comment = f"Spark co R\u00b2 cao hon sklearn {r2_abs:.6f}"
    else:
        r2_comment = f"sklearn co R\u00b2 cao hon Spark {r2_abs:.6f}"

    metric_table = "\n".join(metric_rows)
    top_text = "\n".join(top_lines)

    report = (
        "# Phase 4 - So sanh hai mo hinh Random Forest\n\n"
        "## 1. Pham vi\n\n"
        "Phase 4 so sanh ket qua tu:\n\n"
        "- **Phase 2:** Random Forest Regression bang scikit-learn.\n"
        "- **Phase 3:** RandomForestRegressor bang PySpark Spark MLlib.\n"
        "- Hai phase dung cung tap train/validation/test va cung target `power_output`.\n"
        f"- File prediction duoc doi chieu theo `record_id`; actual cua hai mo hinh khop: **{same_actual}**.\n\n"
        "## 2. So sanh do chinh xac tren test\n\n"
        "| Metric | scikit-learn | Spark MLlib | Spark - sklearn | Ket luan |\n"
        "|---|---:|---:|---:|---|\n"
        f"{metric_table}\n\n"
        f"- {rmse_comment}.\n"
        f"- {mae_comment}.\n"
        f"- {r2_comment}.\n"
        "- Cac chenh lech deu rat nho; khong nen ket luan mot mo hinh vuot troi tuyet doi chi dua tren accuracy.\n\n"
        "## 3. Overfitting va kha nang tong quat hoa\n\n"
        "| Model | Train RMSE | Test RMSE | Gap Test - Train |\n"
        "|---|---:|---:|---:|\n"
        f"| scikit-learn | {p2_train['RMSE']:.6f} | {p2['RMSE']:.6f} | {p2_gap:.6f} |\n"
        f"| Spark MLlib | {p3_train['RMSE']:.6f} | {p3['RMSE']:.6f} | {p3_gap:.6f} |\n\n"
        "Spark co khoang cach train-test nho hon trong output hien tai, cho thay ket qua it bi overfit hon. "
        "Tuy nhien, hai pipeline khong hoan toan dong nhat ve tham so: sklearn dung `min_samples_leaf`, "
        "con Spark khong co tham so tuong duong trong script.\n\n"
        "## 4. Cau hinh va thoi gian\n\n"
        f"{summary_rows}\n\n"
        f"- scikit-learn thu **{int(summary.iloc[0]['tuning_configurations'])}** cau hinh; "
        f"Spark thu **{int(summary.iloc[1]['tuning_configurations'])}** cau hinh.\n"
        "- Spark chay `local[*]` tren mot may; ket qua nay chi danh gia workflow, "
        "chua duoc danh gia kha nang mo rong tren cluster phan tan thuc su.\n"
        f"- Thoi gian pipeline Spark ghi nhan trong `spark_run_info.txt` la "
        f"**{summary.iloc[1]['pipeline_time_s']:.2f}s**. "
        f"Thoi gian train config tot nhat la **{summary.iloc[1]['best_train_time_s']:.2f}s**; "
        f"sklearn la **{summary.iloc[0]['best_train_time_s']:.2f}s**.\n\n"
        "## 5. Feature importance\n\n"
        "Bang importance duoc doc tu **toan bo cac dac trung encoded**:\n\n"
        "- **Phase 2:** `output_phase2/feature_importance_full.csv` (80 dac trung encoded day du).\n"
        "- **Phase 3:** `output_phase3/feature_importance.csv` (toan bo cac dac trung da xuat).\n\n"
        "Importance cua cac cot one-hot da duoc **cong ve 13 dac trung goc** de so sanh cong bang giua hai mo hinh:\n\n"
        f"- Tong importance Phase 2 (scikit-learn): **{p2_total:.6f}**\n"
        f"- Tong importance Phase 3 (Spark MLlib): **{p3_total:.6f}**\n\n"
        "Top 5 dac trung quan trong nhat (tren 13 dac trung goc):\n\n"
        f"{top_text}\n\n"
        "Ca hai mo hinh deu cho thay `wind_speed` la feature quan trong nhat, nhung muc do khac nhau "
        "do cau hinh pipeline va bo tham so khac nhau (sklearn dung `min_samples_leaf`; Spark khong co). "
        "Feature importance cua Random Forest phan anh muc do dong gop theo cay, "
        "khong chung minh quan he nhan-qua.\n\n"
        "## 6. So sanh prediction cung mau\n\n"
        f"- So mau test duoc doi chieu: **{len(predictions):,}**.\n"
        f"- Chenh lech prediction Spark - sklearn: mean = **{pred_diff.mean():.6f}**, "
        f"mean absolute = **{pred_diff.abs().mean():.6f}**.\n"
        "- Ket qua nay giup xac nhan hai mo hinh dang hoc quy luat gan nhau "
        "du khac framework va preprocessing encoding.\n\n"
        "## 7. Ket luan de trinh bay\n\n"
        f"1. **Ve accuracy:** {rmse_comment}; {mae_comment}. "
        "Chenh lech nho, hai mo hinh gan nhu tuong duong ve chat luong du doan.\n"
        "2. **Ve Big Data:** Spark la lua chon phu hop de trien khai tren du lieu lon/cluster "
        "nho pipeline MLlib va kha nang phan tan. "
        "Ket qua `local[*]` hien tai chi danh gia workflow, chua chung minh loi the cluster phan tan.\n"
        "3. **Ve mo hinh:** Ca hai deu phu thuoc manh vao `wind_speed`; co the cai thien bang "
        "feature engineering theo vat ly tuabin, xu ly zero-output, tuning dong nhat hon va cross-validation.\n"
        "4. **Lua chon de xuat:** Chon Spark cho he thong Big Data production; "
        "chon scikit-learn cho prototype nhanh tren mot may va kiem thu mo hinh.\n\n"
        "## 8. Tep ket qua\n\n"
        "- `output_phase4/comparison_metrics.csv`\n"
        "- `output_phase4/model_summary.csv`\n"
        "- `output_phase4/feature_importance_comparison.csv`\n"
        "- `output_phase4/prediction_comparison.csv`\n"
        "- `pics/phase4/metrics_comparison.png`\n"
        "- `pics/phase4/prediction_comparison.png`\n"
        "- `pics/phase4/feature_importance_comparison.png`\n"
    )
    (OUT / "comparison_report.md").write_text(report, encoding="utf-8")


def main():
    print("PHASE 4 - MODEL COMPARISON")
    metrics = load_metrics()
    predictions = load_predictions()
    features = load_feature_importance()
    summary = load_model_summary()

    # Kiem tra nhanh feature importance
    print("\n--- Kiem tra feature importance ---")
    for model_name in MODELS:
        sub = features[features["model"] == model_name]
        n_feat = len(sub)
        total_imp = sub["importance"].sum()
        print(f"  {model_name}: {n_feat} feature goc, tong importance = {total_imp:.6f}")

    metrics.to_csv(OUT / "comparison_metrics.csv", index=False)
    summary.to_csv(OUT / "model_summary.csv", index=False)
    features.to_csv(OUT / "feature_importance_comparison.csv", index=False)
    predictions.to_csv(OUT / "prediction_comparison.csv", index=False)

    make_metric_plot(metrics)
    make_prediction_plot(predictions)
    make_feature_plot(features)
    make_report(metrics, summary, features, predictions)

    print(f"\nMetrics: {OUT / 'comparison_metrics.csv'}")
    print(f"Report:  {OUT / 'comparison_report.md'}")
    print(f"Plots:   {PICS}")
    print("Done.")


if __name__ == "__main__":
    main()
