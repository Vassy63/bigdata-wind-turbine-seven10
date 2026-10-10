# Phase 4 - So sanh hai mo hinh Random Forest

## 1. Pham vi

Phase 4 so sanh ket qua tu:

- **Phase 2:** Random Forest Regression bang scikit-learn.
- **Phase 3:** RandomForestRegressor bang PySpark Spark MLlib.
- Hai phase dung cung tap train/validation/test va cung target `power_output`.
- File prediction duoc doi chieu theo `record_id`; actual cua hai mo hinh khop: **True**.

## 2. So sanh do chinh xac tren test

| Metric | scikit-learn | Spark MLlib | Spark - sklearn | Ket luan |
|---|---:|---:|---:|---|
| MAE | 12.587087 | 12.613750 | +0.026663 | Spark kem hon |
| RMSE | 15.896208 | 15.843621 | -0.052586 | Spark tot hon |
| R2 | 0.491999 | 0.495354 | +0.003355 | Spark tot hon |

- Spark co RMSE nho hon sklearn 0.0526 (0.33% tuong doi so voi sklearn).
- sklearn co MAE nho hon Spark 0.0267 (0.21% tuong doi).
- Spark co R² cao hon sklearn 0.003355.
- Cac chenh lech deu rat nho; khong nen ket luan mot mo hinh vuot troi tuyet doi chi dua tren accuracy.

## 3. Overfitting va kha nang tong quat hoa

| Model | Train RMSE | Test RMSE | Gap Test - Train |
|---|---:|---:|---:|
| scikit-learn | 11.107241 | 15.896208 | 4.788966 |
| Spark MLlib | 14.854792 | 15.843621 | 0.988829 |

Spark co khoang cach train-test nho hon trong output hien tai, cho thay ket qua it bi overfit hon. Tuy nhien, hai pipeline khong hoan toan dong nhat ve tham so: sklearn dung `min_samples_leaf`, con Spark khong co tham so tuong duong trong script.

## 4. Cau hinh va thoi gian

| model                  | framework           | best_configuration                                 |   tuning_configurations |   best_validation_rmse |   best_train_time_s |   pipeline_time_s | execution_mode     |
|:-----------------------|:--------------------|:---------------------------------------------------|------------------------:|-----------------------:|--------------------:|------------------:|:-------------------|
| Phase 2 - scikit-learn | scikit-learn        | n_estimators=200, max_depth=20, min_samples_leaf=2 |                       8 |                15.8253 |               97.2  |            nan    | local scikit-learn |
| Phase 3 - Spark MLlib  | PySpark Spark MLlib | numTrees=200, maxDepth=12                          |                       4 |                15.7719 |              141.42 |            500.39 | local[*]           |

- scikit-learn thu **8** cau hinh; Spark thu **4** cau hinh.
- Spark chay `local[*]` tren mot may; ket qua nay chi danh gia workflow, chua duoc danh gia kha nang mo rong tren cluster phan tan thuc su.
- Thoi gian pipeline Spark ghi nhan trong `spark_run_info.txt` la **500.39s**. Thoi gian train config tot nhat la **141.42s**; sklearn la **97.20s**.

## 5. Feature importance

Bang importance duoc doc tu **toan bo cac dac trung encoded**:

- **Phase 2:** `output_phase2/feature_importance_full.csv` (80 dac trung encoded day du).
- **Phase 3:** `output_phase3/feature_importance.csv` (toan bo cac dac trung da xuat).

Importance cua cac cot one-hot da duoc **cong ve 13 dac trung goc** de so sanh cong bang giua hai mo hinh:

- Tong importance Phase 2 (scikit-learn): **1.000000**
- Tong importance Phase 3 (Spark MLlib): **1.000000**

Top 5 dac trung quan trong nhat (tren 13 dac trung goc):

- **Phase 2 - scikit-learn:** wind_speed (0.6818), air_density (0.0359), blade_angle (0.0355), generator_efficiency (0.0344), altitude (0.0320)
- **Phase 3 - Spark MLlib:** wind_speed (0.8907), turbine_id (0.0266), air_density (0.0113), blade_angle (0.0103), generator_efficiency (0.0100)

Ca hai mo hinh deu cho thay `wind_speed` la feature quan trong nhat, nhung muc do khac nhau do cau hinh pipeline va bo tham so khac nhau (sklearn dung `min_samples_leaf`; Spark khong co). Feature importance cua Random Forest phan anh muc do dong gop theo cay, khong chung minh quan he nhan-qua.

## 6. So sanh prediction cung mau

- So mau test duoc doi chieu: **28,189**.
- Chenh lech prediction Spark - sklearn: mean = **0.000707**, mean absolute = **1.447530**.
- Ket qua nay giup xac nhan hai mo hinh dang hoc quy luat gan nhau du khac framework va preprocessing encoding.

## 7. Ket luan de trinh bay

1. **Ve accuracy:** Spark co RMSE nho hon sklearn 0.0526 (0.33% tuong doi so voi sklearn); sklearn co MAE nho hon Spark 0.0267 (0.21% tuong doi). Chenh lech nho, hai mo hinh gan nhu tuong duong ve chat luong du doan.
2. **Ve Big Data:** Spark la lua chon phu hop de trien khai tren du lieu lon/cluster nho pipeline MLlib va kha nang phan tan. Ket qua `local[*]` hien tai chi danh gia workflow, chua chung minh loi the cluster phan tan.
3. **Ve mo hinh:** Ca hai deu phu thuoc manh vao `wind_speed`; co the cai thien bang feature engineering theo vat ly tuabin, xu ly zero-output, tuning dong nhat hon va cross-validation.
4. **Lua chon de xuat:** Chon Spark cho he thong Big Data production; chon scikit-learn cho prototype nhanh tren mot may va kiem thu mo hinh.

## 8. Tep ket qua

- `output_phase4/comparison_metrics.csv`
- `output_phase4/model_summary.csv`
- `output_phase4/feature_importance_comparison.csv`
- `output_phase4/prediction_comparison.csv`
- `pics/phase4/metrics_comparison.png`
- `pics/phase4/prediction_comparison.png`
- `pics/phase4/feature_importance_comparison.png`
