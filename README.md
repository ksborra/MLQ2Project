# MLQ2Project
Project Description
The motivation behind this study stems from the limitations of the Random Forest algorithm, which treats all attributes equally when constructing decision trees, potentially leading to suboptimal performance on datasets with attributes of varying importance. To address this, we proposed an enhanced Random Forest implementation that dynamically adjusts feature importance by integrating five distinct feature weighting metrics: Shapely Additive Explanations (SHAP) values, mutual information, permutation importance, Pearson correlation, and variance. The model employs a weighted scoring system for these metrics, with SHAP contributing the most to attribute weighting. Furthermore, we implemented periodic updates to feature weights using a decaying average and optimized hyperparameters via Optuna's Tree-structured Parzen Estimator (TPE). The final model configuration, consisting of 84 trees and a maximum depth of 25, demonstrated improved prediction accuracy and faster convergence. This methodology emphasizes the critical role of tailored attribute weighting in enhancing machine learning models' performance while reducing computation overhead.


How to Run Code:
Read in desired csv files (nursery is included in this pacakge) and our custom random forest algorithm will run on it
