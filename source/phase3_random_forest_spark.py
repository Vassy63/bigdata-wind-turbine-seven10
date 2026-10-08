"""
Phase 3: Random Forest Regression with PySpark / Spark MLlib
Người thực hiện: Người 3 (độc lập với Phase 1 và Phase 2)

Mục tiêu:
- Sử dụng PySpark MLlib RandomForestRegressor
- Đọc data/train.csv, data/validation.csv, data/test.csv
- Preprocessing: StringIndexer, OneHotEncoder, StandardScaler, VectorAssembler
- Hyperparameter tuning trên validation
- Đánh giá cuối cùng trên test
- Xuất kết quả vào output_phase3/
"""

import sys
import time
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

ROOT = Path(__file__).resolve().parent.parent
PICS_DIR = ROOT / "pics" / "phase3"

# ============================================================
# 1. KHỞI TẠO SPARK SESSION
# ============================================================
def init_spark():
    """Khởi tạo Spark session với cấu hình memory phù hợp cho RandomForestRegressor."""
    spark = SparkSession.builder \
        .appName("Phase3_RandomForest_WindTurbine") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.shuffle.partitions", "16") \
        .config("spark.default.parallelism", "16") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.memory.fraction", "0.6") \
        .config("spark.memory.storageFraction", "0.3") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    return spark


# ============================================================
# 2. ĐỌC DỮ LIỆU VÀ GHÉP RECORD_ID
# ============================================================
def load_data_with_record_ids(spark, split_name):
    """
    Đọc CSV data và record_ids, ghép chúng lại theo thứ tự dòng.

    Cách tiếp cận an toàn:
    - Đọc cả data và record_ids bằng pandas
    - Ghép theo index (đảm bảo thứ tự dòng khớp)
    - Convert sang Spark DataFrame với record_id đã có sẵn

    Args:
        spark: SparkSession
        split_name: 'train', 'validation', hoặc 'test'

    Returns:
        Spark DataFrame với cột record_id
    """
    data_path = f"data/{split_name}.csv"
    ids_path = f"data/{split_name}_record_ids.csv"

    # Đọc data và record_ids bằng pandas
    df_data = pd.read_csv(data_path)
    df_ids = pd.read_csv(ids_path)

    # Kiểm tra số dòng, header và tính duy nhất trước khi ghép theo index.
    assert list(df_ids.columns) == ["record_id"], (
        f"{split_name}: record_ids phải có đúng cột record_id"
    )
    assert len(df_data) == len(df_ids), f"{split_name}: Số dòng data và record_ids không khớp!"
    assert df_ids["record_id"].notna().all(), f"{split_name}: record_id chứa null!"
    assert df_ids["record_id"].is_unique, f"{split_name}: record_id bị trùng!"

    # ID được ghép trước khi vào Spark; mọi shuffle sau đó vẫn giữ ID trong cùng row.
    df_data.insert(0, "record_id", df_ids["record_id"].astype("int64"))

    # Convert sang Spark DataFrame
    spark_df = spark.createDataFrame(df_data)

    return spark_df


# ============================================================
# 3. KIỂM TRA DỮ LIỆU
# ============================================================
def inspect_data(df, name):
    """In thông tin kiểm tra về DataFrame"""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    print(f"Số dòng: {df.count()}")
    print(f"Số cột: {len(df.columns)}")
    print(f"Schema:")
    df.printSchema()
    print(f"\nSample (3 dòng đầu):")
    df.show(3, truncate=True)

    # Kiểm tra null từng cột (dùng spark_sum tránh xung đột với sum của Python)
    null_counts = df.select([
        (col(c).isNull().cast("long")).alias(c) for c in df.columns
    ]).agg(*[spark_sum(col(c)).alias(c) for c in df.columns]).collect()[0].asDict()

    has_nulls = any(v > 0 for v in null_counts.values())
    if has_nulls:
        print(f"\nCảnh báo: Phát hiện giá trị null:")
        for col_name, null_count in null_counts.items():
            if null_count > 0:
                print(f"  {col_name}: {null_count} nulls")
    else:
        print(f"\nKhông có giá trị null.")


# ============================================================
# 4. PREPROCESSING PIPELINE
# ============================================================
def build_preprocessing_pipeline(categorical_cols, numeric_cols):
    """
    Xây dựng pipeline preprocessing cho Spark MLlib.

    Các bước:
    1. StringIndexer: chuyển categorical thành index
    2. OneHotEncoder: one-hot encoding các categorical
    3. VectorAssembler: gộp numeric + encoded categorical
    4. StandardScaler: chuẩn hóa features vector

    Args:
        categorical_cols: danh sách tên cột categorical
        numeric_cols: danh sách tên cột numeric

    Returns:
        Pipeline các transformer
    """
    stages = []

    # Stage 1: StringIndexer cho từng categorical column
    indexers = []
    indexed_cols = []
    for cat_col in categorical_cols:
        indexer = StringIndexer(
            inputCol=cat_col,
            outputCol=f"{cat_col}_indexed",
            handleInvalid="keep"  # giữ unseen categories
        )
        indexers.append(indexer)
        indexed_cols.append(f"{cat_col}_indexed")
    stages.extend(indexers)

    # Stage 2: OneHotEncoder cho tất cả indexed columns
    encoder = OneHotEncoder(
        inputCols=indexed_cols,
        outputCols=[f"{col}_encoded" for col in indexed_cols],
        handleInvalid="keep"
    )
    stages.append(encoder)
    encoded_cols = [f"{col}_encoded" for col in indexed_cols]

    # Stage 3: VectorAssembler gộp numeric + encoded
    assembler_inputs = numeric_cols + encoded_cols
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features_unscaled"
    )
    stages.append(assembler)

    # Stage 4: StandardScaler
    scaler = StandardScaler(
        inputCol="features_unscaled",
        outputCol="features",
        withMean=True,
        withStd=True
    )
    stages.append(scaler)

    pipeline = Pipeline(stages=stages)
    return pipeline


# ============================================================
# 5. TRAIN MODEL
# ============================================================
def train_random_forest(train_df, num_trees, max_depth, seed=42):
    """
    Train Random Forest model với cấu hình cho trước.

    Args:
        train_df: Spark DataFrame đã preprocessing với cột 'features' và 'power_output'
        num_trees: số cây
        max_depth: độ sâu tối đa
        seed: random seed

    Returns:
        Trained model
    """
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="power_output",
        numTrees=num_trees,
        maxDepth=max_depth,
        seed=seed
    )

    model = rf.fit(train_df)
    return model


# ============================================================
# 6. EVALUATION
# ============================================================
def evaluate_model(predictions_df, metric_name="rmse"):
    """
    Tính metric cho predictions.

    Args:
        predictions_df: DataFrame với cột 'power_output' và 'prediction'
        metric_name: 'rmse', 'mae', hoặc 'r2'

    Returns:
        metric value
    """
    evaluator = RegressionEvaluator(
        labelCol="power_output",
        predictionCol="prediction",
        metricName=metric_name
    )
    return evaluator.evaluate(predictions_df)


def compute_all_metrics(predictions_df):
    """Tính tất cả metrics: MAE, RMSE, R²"""
    mae = evaluate_model(predictions_df, "mae")
    rmse = evaluate_model(predictions_df, "rmse")
    r2 = evaluate_model(predictions_df, "r2")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def get_feature_names_from_metadata(transformed_df, num_features):
    """Lấy tên attribute theo idx từ metadata của vector sau Pipeline."""
    # StandardScaler có thể không giữ lại ml_attr đầy đủ; assembler output vẫn giữ.
    for column_name in ("features", "features_unscaled"):
        field = transformed_df.schema[column_name]
        ml_attr = field.metadata.get("ml_attr", {})
        attrs = ml_attr.get("attrs", {})
        idx_to_name = {}
        for attr_group in attrs.values():
            for item in attr_group:
                idx_to_name[int(item["idx"])] = item["name"]
        if set(idx_to_name) == set(range(num_features)):
            return [idx_to_name[i] for i in range(num_features)]
    return None


def get_feature_names_from_input_metadata(transformed_df, assembler_inputs, num_features):
    """Fallback đọc Attribute metadata từng input của VectorAssembler."""
    names = []
    for column_name in assembler_inputs:
        field = transformed_df.schema[column_name]
        ml_attr = field.metadata.get("ml_attr", {})
        attrs = ml_attr.get("attrs", {})
        indexed = []
        for attr_group in attrs.values():
            indexed.extend((int(item["idx"]), item["name"]) for item in attr_group)
        if indexed:
            names.extend(name for _, name in sorted(indexed))
        else:
            names.append(column_name)
    return names if len(names) == num_features else None


def build_feature_names_from_pipeline(preprocessing_model, numeric_cols, categorical_cols):
    """Dựng tên feature theo cấu trúc VectorAssembler, dùng labels của StringIndexer."""
    indexer_models = {}
    for stage in preprocessing_model.stages:
        if stage.__class__.__name__ == "StringIndexerModel":
            indexer_models[stage.getInputCol()] = stage

    encoder_model = None
    for stage in preprocessing_model.stages:
        if stage.__class__.__name__ == "OneHotEncoderModel":
            encoder_model = stage
            break
    if encoder_model is None or len(encoder_model.getInputCols()) != len(categorical_cols):
        return None

    encoded_names = []
    for cat_idx, cat_col in enumerate(categorical_cols):
        indexer_model = indexer_models.get(cat_col)
        if indexer_model is None:
            return None
        labels = indexer_model.labels
        # OneHotEncoder mặc định dropLast=True; categorySizes là số category
        # trước khi bỏ một cột đại diện.
        size = max(int(encoder_model.categorySizes[cat_idx]) - 1, 0)
        for i in range(size):
            label = labels[i] if i < len(labels) else str(i)
            encoded_names.append(f"{cat_col}_{label}")

    return list(numeric_cols) + encoded_names


def validate_prediction_output(predictions_df, source_df, split_name):
    """Kiểm tra đủ ID, không trùng và actual khớp dữ liệu split ban đầu."""
    expected = source_df.select("record_id", "power_output").toPandas()
    actual = predictions_df.select("record_id", "power_output", "prediction").toPandas()
    assert len(actual) == len(expected), f"{split_name}: mất record khi tạo prediction"
    assert actual["record_id"].is_unique, f"{split_name}: duplicate record_id"
    assert actual["record_id"].notna().all(), f"{split_name}: record_id null"
    expected = expected.sort_values("record_id").reset_index(drop=True)
    actual_check = actual[["record_id", "power_output"]].sort_values("record_id").reset_index(drop=True)
    assert expected["record_id"].equals(actual_check["record_id"]), (
        f"{split_name}: record_id không khớp input"
    )
    assert (expected["power_output"] == actual_check["power_output"]).all(), (
        f"{split_name}: actual_power_output không khớp input"
    )
    return actual


# ============================================================
# 7. MAIN EXECUTION
# ============================================================
def main():
    pipeline_start = time.perf_counter()
    print("="*70)
    print("PHASE 3: RANDOM FOREST REGRESSION WITH PYSPARK")
    print("="*70)

    # Khởi tạo Spark
    spark = init_spark()
    print(f"\nSpark Version: {spark.version}")
    print(f"Spark Master: {spark.sparkContext.master}")
    print(f"App Name: {spark.sparkContext.appName}")
    print(f"Default Parallelism: {spark.sparkContext.defaultParallelism}")

    # ============================================================
    # BƯỚC 1: ĐỌC DỮ LIỆU
    # ============================================================
    print("\n" + "="*70)
    print("BƯỚC 1: ĐỌC DỮ LIỆU")
    print("="*70)

    train_df = load_data_with_record_ids(spark, "train")
    val_df = load_data_with_record_ids(spark, "validation")
    test_df = load_data_with_record_ids(spark, "test")

    inspect_data(train_df, "TRAIN")
    inspect_data(val_df, "VALIDATION")
    inspect_data(test_df, "TEST")

    print(f"\nTrain partitions: {train_df.rdd.getNumPartitions()}")
    print(f"Validation partitions: {val_df.rdd.getNumPartitions()}")
    print(f"Test partitions: {test_df.rdd.getNumPartitions()}")

    # ============================================================
    # BƯỚC 2: ĐỊNH NGHĨA FEATURES
    # ============================================================
    print("\n" + "="*70)
    print("BƯỚC 2: ĐỊNH NGHĨA FEATURES")
    print("="*70)

    categorical_cols = ["region", "turbine_id", "maintenance_status", "technician_note"]
    numeric_cols = ["wind_speed", "wind_direction", "air_density", "temperature",
                    "blade_angle", "rotor_speed", "generator_efficiency",
                    "altitude", "humidity"]
    target_col = "power_output"
    assembler_inputs = numeric_cols + [f"{col}_indexed_encoded" for col in categorical_cols]

    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Target: {target_col}")

    # ============================================================
    # BƯỚC 3: PREPROCESSING
    # ============================================================
    print("\n" + "="*70)
    print("BƯỚC 3: PREPROCESSING")
    print("="*70)

    preprocessing_pipeline = build_preprocessing_pipeline(categorical_cols, numeric_cols)
    print("Pipeline stages:")
    for i, stage in enumerate(preprocessing_pipeline.getStages()):
        print(f"  {i+1}. {stage.__class__.__name__}")

    print("\nFitting preprocessing pipeline on TRAIN data...")
    start_fit = time.perf_counter()
    preprocessing_model = preprocessing_pipeline.fit(train_df)
    fit_time = time.perf_counter() - start_fit
    print(f"Preprocessing fit time: {fit_time:.2f}s")

    print("\nTransforming datasets...")
    start_transform = time.perf_counter()
    # Chỉ persist train (dùng lại nhiều lần: baseline + từng config tuning + final).
    # Dùng MEMORY_AND_DISK để tránh OOM khi bộ nhớ không đủ giữ toàn bộ.
    train_transformed = preprocessing_model.transform(train_df).persist(StorageLevel.MEMORY_AND_DISK)
    train_count = train_transformed.count()
    transform_time = time.perf_counter() - start_transform
    print(f"Train transformed & cached: {train_count} rows in {transform_time:.2f}s")

    # Validation/test chỉ dùng theo từng lượt -> transform lazy, không cache để tiết kiệm JVM heap.
    val_transformed = preprocessing_model.transform(val_df)
    test_transformed = preprocessing_model.transform(test_df)
    print("Validation transformed (lazy, not cached)")
    print("Test transformed (lazy, not cached)")

    # In cấu hình Spark thực tế đang áp dụng (conf có thể bị bỏ qua nếu JVM đã khởi động).
    print(f"\nSpark master: {spark.sparkContext.master}")
    print(f"driver.memory (conf): {spark.sparkContext.getConf().get('spark.driver.memory', 'N/A')}")
    print(f"executor.memory (conf): {spark.sparkContext.getConf().get('spark.executor.memory', 'N/A')}")
    print(f"sql.shuffle.partitions: {spark.conf.get('spark.sql.shuffle.partitions', 'N/A')}")
    print(f"default.parallelism: {spark.sparkContext.defaultParallelism}")
    print(f"JVM max heap (driver): {spark.sparkContext._jvm.java.lang.Runtime.getRuntime().maxMemory() / (1024**3):.2f} GB")

    # ============================================================
    # BƯỚC 4: BASELINE MODEL
    # ============================================================
    print("\n" + "="*70)
    print("BƯỚC 4: BASELINE MODEL")
    print("="*70)

    baseline_config = {"numTrees": 50, "maxDepth": 8}
    print(f"Baseline config: {baseline_config}")

    start_train = time.perf_counter()
    baseline_model = train_random_forest(
        train_transformed,
        num_trees=baseline_config["numTrees"],
        max_depth=baseline_config["maxDepth"],
        seed=42
    )
    baseline_train_time = time.perf_counter() - start_train
    print(f"Baseline training time: {baseline_train_time:.2f}s")

    # Evaluate baseline
    train_pred_baseline = baseline_model.transform(train_transformed)
    val_pred_baseline = baseline_model.transform(val_transformed)

    train_metrics_baseline = compute_all_metrics(train_pred_baseline)
    val_metrics_baseline = compute_all_metrics(val_pred_baseline)

    print(f"\nBaseline Train Metrics:")
    for k, v in train_metrics_baseline.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nBaseline Validation Metrics:")
    for k, v in val_metrics_baseline.items():
        print(f"  {k}: {v:.4f}")

    if "--baseline-only" in sys.argv:
        print("\n--baseline-only mode: stopping after baseline. No tuning/test run.")
        train_transformed.unpersist()
        spark.stop()
        return

    # ============================================================
    # BƯỚC 5: HYPERPARAMETER TUNING
    # ============================================================
    print("\n" + "="*70)
    print("BƯỚC 5: HYPERPARAMETER TUNING")
    print("="*70)

    param_grid = [
        {"numTrees": 50, "maxDepth": 8},
        {"numTrees": 100, "maxDepth": 8},
        {"numTrees": 100, "maxDepth": 12},
        {"numTrees": 200, "maxDepth": 12},
    ]

    print(f"Testing {len(param_grid)} configurations on validation set...")

    tuning_results = []
    for i, params in enumerate(param_grid):
        print(f"\nConfig {i+1}/{len(param_grid)}: {params}")

        start = time.perf_counter()
        model = train_random_forest(
            train_transformed,
            num_trees=params["numTrees"],
            max_depth=params["maxDepth"],
            seed=42
        )
        fit_time_config = time.perf_counter() - start

        val_pred = model.transform(val_transformed)
        val_metrics = compute_all_metrics(val_pred)

        result = {
            "numTrees": params["numTrees"],
            "maxDepth": params["maxDepth"],
            "validation_MAE": val_metrics["MAE"],
            "validation_RMSE": val_metrics["RMSE"],
            "validation_R2": val_metrics["R2"],
            "fit_time_seconds": fit_time_config
        }
        tuning_results.append(result)

        print(f"  Validation MAE: {val_metrics['MAE']:.4f}")
        print(f"  Validation RMSE: {val_metrics['RMSE']:.4f}")
        print(f"  Validation R²: {val_metrics['R2']:.4f}")
        print(f"  Fit time: {fit_time_config:.2f}s")

    # Chọn config tốt nhất dựa trên validation RMSE
    best_result = min(tuning_results, key=lambda x: x["validation_RMSE"])
    print(f"\n{'='*70}")
    print("BEST CONFIGURATION (lowest validation RMSE):")
    print(f"{'='*70}")
    for k, v in best_result.items():
        print(f"  {k}: {v}")

    # ============================================================
    # BƯỚC 6: FINAL MODEL VÀ TEST EVALUATION
    # ============================================================
    print("\n" + "="*70)
    print("BƯỚC 6: FINAL MODEL VÀ TEST EVALUATION")
    print("="*70)
    print("Test set is used only for final evaluation.")

    print(f"\nTraining final model with best config...")
    start_final = time.perf_counter()
    final_model = train_random_forest(
        train_transformed,
        num_trees=best_result["numTrees"],
        max_depth=best_result["maxDepth"],
        seed=42
    )
    final_train_time = time.perf_counter() - start_final
    print(f"Final model training time: {final_train_time:.2f}s")

    # Predict on all sets
    train_pred_final = final_model.transform(train_transformed)
    val_pred_final = final_model.transform(val_transformed)
    test_pred_final = final_model.transform(test_transformed)

    train_metrics_final = compute_all_metrics(train_pred_final)
    val_metrics_final = compute_all_metrics(val_pred_final)
    test_metrics_final = compute_all_metrics(test_pred_final)

    print(f"\nFinal Model - Train Metrics:")
    for k, v in train_metrics_final.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nFinal Model - Validation Metrics:")
    for k, v in val_metrics_final.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nFinal Model - Test Metrics:")
    for k, v in test_metrics_final.items():
        print(f"  {k}: {v:.4f}")

    # ============================================================
    # BƯỚC 7: FEATURE IMPORTANCE
    # ============================================================
    print("\n" + "="*70)
    print("BƯỚC 7: FEATURE IMPORTANCE")
    print("="*70)

    # Feature importance tương ứng vector sau Pipeline (numeric + OneHotEncoder).
    # test_pred_final và các prediction khác không ảnh hưởng phần này.
    feature_importances = final_model.featureImportances.toArray()
    num_features = len(feature_importances)

    # Không dùng partition order hay thứ tự Stage để suy tên.
    feature_names = get_feature_names_from_metadata(train_transformed, num_features)
    name_source = "vector metadata"

    if feature_names is None:
        feature_names = get_feature_names_from_input_metadata(
            train_transformed, assembler_inputs, num_features
        )
        name_source = "assembler input metadata"

    if feature_names is None:
        # Cuối cùng dựng tên từ labels của StringIndexer theo đúng thứ tự VectorAssembler.
        feature_names = build_feature_names_from_pipeline(
            preprocessing_model, numeric_cols, categorical_cols
        )
        name_source = "StringIndexer labels"
        if feature_names is None:
            raise RuntimeError(
                "Không xác định được tên feature chính xác sau OneHotEncoder; "
                "từ chối ghi feature_importance.csv vì có thể sai ánh xạ."
            )

    assert len(feature_names) == num_features, (
        f"feature_names ({len(feature_names)}) != feature_importances ({num_features})"
    )
    print(f"Total features after encoding: {num_features}")
    print(f"Feature names lấy từ: {name_source}")
    print("Mỗi encoded column của OneHotEncoder được giữ riêng, không gộp.")

    # ============================================================
    # BƯỚC 8: XUẤT OUTPUT
    # ============================================================
    print("\n" + "="*70)
    print("BƯỚC 8: XUẤT OUTPUT")
    print("="*70)

    root = Path(__file__).resolve().parent.parent
    output_dir = root / "output_phase3"
    output_dir.mkdir(exist_ok=True)
    pics_dir = root / "pics" / "phase3"
    pics_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")

    # 8.1: Metrics summary
    metrics_data = {
        "split": ["train", "validation", "test"],
        "MAE": [train_metrics_final["MAE"], val_metrics_final["MAE"], test_metrics_final["MAE"]],
        "RMSE": [train_metrics_final["RMSE"], val_metrics_final["RMSE"], test_metrics_final["RMSE"]],
        "R2": [train_metrics_final["R2"], val_metrics_final["R2"], test_metrics_final["R2"]]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    print(f"✓ Đã lưu: {output_dir / 'metrics.csv'}")

    # 8.2: Hyperparameter tuning results
    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(output_dir / "param_search_results.csv", index=False)
    print(f"✓ Đã lưu: {output_dir / 'param_search_results.csv'}")

    # 8.3: Feature importance
    importance_data = {
        "feature_name": feature_names,
        "importance": feature_importances
    }
    importance_df = pd.DataFrame(importance_data)
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    print(f"✓ Đã lưu: {output_dir / 'feature_importance.csv'}")
    print(f"  Top 10 features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"    {row['feature_name']:40s} {row['importance']:.6f}")

    # 8.4: Test predictions
    test_pred_pd = validate_prediction_output(test_pred_final, test_df, "test")
    test_pred_pd = test_pred_pd.rename(columns={
        "power_output": "actual_power_output",
        "prediction": "predicted_power_output",
    })
    test_pred_pd.to_csv(output_dir / "test_predictions.csv", index=False)
    print(f"✓ Đã lưu: {output_dir / 'test_predictions.csv'}")
    print(f"  Test predictions: {len(test_pred_pd)} rows")
    print(f"  ✓ record_id và actual_power_output đã đối chiếu với input")

    # 8.5: Validation predictions
    val_pred_pd = validate_prediction_output(val_pred_final, val_df, "validation")
    val_pred_pd = val_pred_pd.rename(columns={
        "power_output": "actual_power_output",
        "prediction": "predicted_power_output",
    })
    val_pred_pd.to_csv(output_dir / "validation_predictions.csv", index=False)
    print(f"✓ Đã lưu: {output_dir / 'validation_predictions.csv'}")
    print(f"  Validation predictions: {len(val_pred_pd)} rows")
    print(f"  ✓ record_id và actual_power_output đã đối chiếu với input")

    # 8.6: Spark run info
    # Đo tổng thời gian chạy toàn bộ pipeline (Spark init -> ghi file output)
    pipeline_time = time.perf_counter() - pipeline_start

    # Lấy thông tin executor và worker nếu có
    sc = spark.sparkContext
    spark_master = sc.master
    is_local = spark_master.startswith("local")

    if is_local:
        mode_info = f"Local mode: {spark_master}"
        executor_info = "N/A (local mode)"
    else:
        mode_info = f"Cluster mode: {spark_master}"
        # Thử lấy thông tin executor nếu chạy cluster
        try:
            executor_info = f"Executors: {sc.getExecutorMemoryStatus().keys()}"
        except:
            executor_info = "Cannot retrieve executor info"

    spark_info = f"""SPARK CONFIGURATION
==================
Spark Version: {spark.version}
Spark Master: {spark_master}
Mode: {mode_info}
App Name: {spark.sparkContext.appName}
Default Parallelism: {spark.sparkContext.defaultParallelism}
Executor/Worker Info: {executor_info}

DATASET PARTITIONS
==================
Train partitions: {train_df.rdd.getNumPartitions()}
Validation partitions: {val_df.rdd.getNumPartitions()}
Test partitions: {test_df.rdd.getNumPartitions()}

TIMING
======
Preprocessing fit time: {fit_time:.2f}s
Final model training time: {final_train_time:.2f}s
Whole pipeline execution time: {pipeline_time:.2f}s

BEST MODEL CONFIG
=================
numTrees: {best_result['numTrees']}
maxDepth: {best_result['maxDepth']}
seed: 42

FINAL METRICS
=============
Train MAE: {train_metrics_final['MAE']:.4f}
Train RMSE: {train_metrics_final['RMSE']:.4f}
Train R²: {train_metrics_final['R2']:.4f}

Validation MAE: {val_metrics_final['MAE']:.4f}
Validation RMSE: {val_metrics_final['RMSE']:.4f}
Validation R²: {val_metrics_final['R2']:.4f}

Test MAE: {test_metrics_final['MAE']:.4f}
Test RMSE: {test_metrics_final['RMSE']:.4f}
Test R²: {test_metrics_final['R2']:.4f}
"""

    with open(output_dir / "spark_run_info.txt", "w", encoding="utf-8") as f:
        f.write(spark_info)
    print(f"✓ Đã lưu: {output_dir / 'spark_run_info.txt'}")

    # ============================================================
    # 9. BIỂU ĐỒ
    # ============================================================
    print("\n[9/9] Tạo biểu đồ ...")

    top10 = importance_df.sort_values("importance", ascending=False).head(10)
    top10 = top10.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(14, 9))
    bars = ax.barh(top10["feature_name"], top10["importance"], color="#2f6690")
    ax.set_title("Phase 3 Random Forest: Top 10 Feature Importance", fontsize=18, pad=16)
    ax.set_xlabel("Importance", fontsize=13)
    ax.set_ylabel("Feature", fontsize=13)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, top10["importance"]):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.6f}", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(pics_dir / "rf_spark_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    configurations = [(50, 8), (100, 8), (100, 12), (200, 12)]
    selected = tuning_df.set_index(["numTrees", "maxDepth"]).loc[configurations].reset_index()
    labels = [f"{trees} trees\ndepth {depth}" for trees, depth in configurations]
    best_index = labels.index("200 trees\ndepth 12")
    fig, ax = plt.subplots(figsize=(13, 8))
    colors = ["#7aa6c2"] * len(selected)
    colors[best_index] = "#d95f02"
    bars = ax.bar(labels, selected["validation_RMSE"], color=colors, width=0.62)
    ax.set_title("Phase 3 Random Forest: Validation RMSE by Configuration", fontsize=18, pad=16)
    ax.set_xlabel("Random Forest configuration", fontsize=13)
    ax.set_ylabel("Validation RMSE", fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for index, (bar, value) in enumerate(zip(bars, selected["validation_RMSE"])):
        label = f"{value:.6f}"
        if index == best_index:
            label += "\nBest"
        ax.text(bar.get_x() + bar.get_width() / 2, value, label, ha="center", va="bottom", fontsize=11, fontweight="bold" if index == best_index else "normal")
    fig.tight_layout()
    fig.savefig(pics_dir / "rf_spark_param_search.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    metrics_plot = metrics_df.copy()
    metrics_plot["split"] = metrics_plot["split"].str.capitalize()
    metrics_plot = metrics_plot.set_index("split").loc[["Train", "Validation", "Test"]].reset_index()
    metrics = ["MAE", "RMSE", "R2"]
    metric_labels = ["MAE", "RMSE", "R²"]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    x_positions = range(len(metrics_plot))
    width = 0.24
    fig, ax = plt.subplots(figsize=(13, 8))
    for offset, metric, label, color in zip((-width, 0, width), metrics, metric_labels, colors):
        values = metrics_plot[metric]
        bars = ax.bar([position + offset for position in x_positions], values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9, rotation=90)
    ax.set_title("Phase 3 Random Forest: Train, Validation, and Test Evaluation", fontsize=18, pad=16)
    ax.set_xlabel("Data split", fontsize=13)
    ax.set_ylabel("Metric value", fontsize=13)
    ax.set_xticks(list(x_positions), metrics_plot["split"])
    ax.legend(title="Metric")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(pics_dir / "rf_spark_test_evaluation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Đã lưu biểu đồ vào: {pics_dir.absolute()}")

    # ============================================================
    # KẾT THÚC
    # ============================================================
    print("\n" + "="*70)
    print("HOÀN THÀNH PHASE 3")
    print("="*70)
    print(f"\nTất cả output đã được lưu vào: {output_dir.absolute()}")
    print(f"Thời gian toàn bộ pipeline: {pipeline_time:.2f}s")
    print(f"\nSpark UI (đang mở khi ứng dụng chạy): http://localhost:4040")
    print(f"Spark session sẽ tự dừng sau khi script kết thúc.")

    train_transformed.unpersist()
    val_transformed.unpersist()
    test_transformed.unpersist()
    spark.stop()
    print("Spark session đã dừng.")


if __name__ == "__main__":
    main()
