# Exoplanet Analysis

## Overview
The goal of this project is to classify exoplanet candidates into three categories:
1. **Confirmed Planets**
2. **False Positives**
3. **Candidates**

Machine learning models like Random Forests and XGBoost are used for this task, with hyperparameter optimization for improved accuracy.

---

## Dataset
The dataset used in this project is the **Cumulative Kepler Data**, available on [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu).

---

## Requirements
Install the necessary Python libraries using:

```bash
pip install -r requirements.txt
```

## How to Use
1. Clone this repository:
``` bash
git clone https://github.com/krishmathrani2/exoplanetary_analysis.git
cd exoplanetary_analysis
``` 
2. Install the required libraries
``` bash
pip install -r requirements.txt
```
3. Place the cumulative.csv dataset in the root directory of the project.
4. Run the script:
``` bash
python main.py
```
5. Explore the output, including the feature importance, confusion matrices, and classification reports.

---

## Models Used
Random Forest Classifier: \
    - Provides initial predictions with feature importance analysis. \
    - Hyperparameter optimization is done using GridSearchCV to improve accuracy.

XGBoost Classifier: \
    - Offers better performance with faster training. \
    - Implements gradient boosting for enhanced prediction accuracy.

---

## Outputs
Feature Importance \
Feature importance is calculated to understand which features contribute most to the classification task.

Confusion Matrices \
Confusion matrices are displayed for both models to visualize their performance.

Accuracy Scores \
Random Forest (Initial): Accuracy before optimization. \
Random Forest (Optimized): Accuracy after hyperparameter tuning. \
XGBoost: Final accuracy score achieved with XGBoost.


---


## Screenshots (Assets)
Random Forest Confusion Matrix

XGBoost Confusion Matrix


---


## Contributions
Feel free to fork this repository and create pull requests to improve the project or add additional models for analysis.


---


## Author
Krish Mathrani
