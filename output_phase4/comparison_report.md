# Phase 4 - So sanh hai mo hinh Random Forest

## 1. Pham vi

Phase 4 so sanh ket qua tu:

- **Phase 2:** Random Forest Regression bang scikit-learn.
- **Phase 3:** RandomForestRegressor bang PySpark Spark MLlib.
- Hai phase dung cung tap train/validation/test va cung target `power_output`.
- File prediction duoc doi chieu theo `record_id`; actual cua hai mo hinh khop: **True**.

## 2. So sanh do chinh xac tren test

| Metric | scikit-learn | Spark MLlib | Spark - sklearn | Ket luan      |
|---     |---:          |---:         |---:             |---            |
| MAE    | 12.587087    | 12.613750   | +0.026663       | Spark kem hon |
| RMSE   | 15.896208    | 15.843621   | -0.052586       | Spark tot hon |
| R2     | 0.491999     | 0.495354    | +0.003355       | Spark tot hon |

- Spark co RMSE nho hon **0.0526** va R² cao hon **+0.0034**.
- scikit-learn co MAE nho hon **0.0267**.
- Cac chenh lech deu rat nho, vi vay khong nen ket luan mot mo hinh vuot troi tuyet doi chi dua tren accuracy.

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
- Spark phu hop hon khi can chay tren cluster va mo rong du lieu; output hien tai chay `local[*]`, khong phai benchmark cluster phan tan.
- Thoi gian pipeline Spark ghi nhan trong `spark_run_info.txt` la **500.39s**. Thoi gian train config tot nhat la **141.42s**; sklearn la **97.20s**.

## 5. Feature importance

Importance da duoc gop tu cac cot one-hot ve feature goc de so sanh cong bang. Luu y: Phase 2 chi xuat top-20 encoded features, con Phase 3 xuat day du cac feature; do do bang nay dung de so sanh thu hang va cac feature da xuat, khong phai tong importance day du cua Phase 2:

- **Phase 2 - scikit-learn:** wind_speed (0.682), air_density (0.036), blade_angle (0.036), generator_efficiency (0.034), altitude (0.032)
- **Phase 3 - Spark MLlib:** wind_speed (0.891), air_density (0.011), blade_angle (0.010), generator_efficiency (0.010), wind_direction (0.008)

Cả hai mo hinh deu cho thay `wind_speed` la feature quan trong nhat. Spark co importance cua `wind_speed` cao hon trong output (xap xi 0.891 so voi 0.682). Luu y Phase 2 chi luu top-20 encoded features, nen khong dung tong importance cua hai file de so sanh truc tiep. Feature importance cua Random Forest la muc do dong gop theo cay, khong phai quan he nhan-qua.

## 6. So sanh prediction cung mau

- So mau test duoc doi chieu: **28,189**.
- Chenh lech prediction Spark - sklearn: mean = **0.000707**, mean absolute = **1.447530**.
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
