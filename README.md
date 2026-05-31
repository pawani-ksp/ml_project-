# Final Machine Learning Project

Using customer retail dataset implement various models and compare them.

## Project Workflow
1. Load dataset, handle missing values, and encode.
2. Standardize features and calculate revenue.
3. Compare Logistic Regression, Decision Tree, and KNN.

## Results

| Model | Test Accuracy | CV Accuracy |
|---|---|---|
| Logistic Regression | 0.9140 | 0.9114 |
| Decision Tree | 0.9777 | 0.9820 |
| KNN (k=19) ⭐ | 0.9927 | 0.9930 |

## Plots
All required plots are available in the `plots/` directory:
- 01_customer_value_distribution.png
- 02_top10_countries_revenue.png
- 03_feature_correlation_heatmap.png
- 04_knn_elbow_curve.png
- 05_confusion_matrices.png
- 06_model_comparison.png
- 07_decision_tree_feature_importance.png
