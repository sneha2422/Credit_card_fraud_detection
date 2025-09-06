### Project Overview 📝

This project is a detailed case study on **credit card fraud detection** using machine learning. It addresses the fundamental challenge of **highly imbalanced datasets** and provides a complete workflow from data preprocessing to model evaluation and optimization. The project is structured within a Jupyter Notebook, making it easy to follow and reproduce the analysis.

***

### Key Features ✨

* **Data Preprocessing:** The project handles a highly skewed dataset by employing **undersampling**, which creates a balanced dataset of 984 transactions (492 legitimate and 492 fraudulent) for effective model training.
* **Model Comparison:** The project compares the performance of three classification models: Logistic Regression, Random Forest, and XGBoost.
* **Performance Evaluation:** Model effectiveness is measured using key metrics suitable for imbalanced data, including **Accuracy**, **Precision**, **Recall**, and the **F1-Score**.
* **Final Visualization:** The project concludes with a **confusion matrix** to visually represent the best model's performance on the test data.

***

### Project Structure 📂

The analysis is organized into a sequential workflow:

1.  **Data Loading & Exploration:** The `creditcard.csv` dataset is loaded and its class distribution is examined.
2.  **Data Balancing:** The dataset is undersampled to create a new, balanced dataset.
3.  **Model Implementation & Evaluation:** Each model is trained on the balanced dataset and evaluated.
4.  **Hyperparameter Tuning:** The XGBoost model is fine-tuned to find its optimal parameters.
5.  **Final Visualization:** A confusion matrix is generated for the best-performing model.

***

### Prerequisites 🛠️

To run the notebook, you need **Python 3.x** and the following libraries:
`numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, and `imbalanced-learn`.

You can install them via pip: `pip install pandas scikit-learn xgboost matplotlib seaborn imbalanced-learn`.

***

### Model Performance 📈

The table below summarizes the key metrics for the models evaluated on the undersampled dataset.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.939 | — | — | — |
| **Random Forest** | 0.929 | **0.988** | 0.867 | **0.924** |
| **XGBoost (Untuned)** | 0.919 | 0.966 | 0.867 | 0.914 |
| **XGBoost (Tuned)** | 0.919 | 0.966 | 0.867 | 0.914 |

The **Random Forest** model provided the highest F1-Score and Precision, making it the most effective model for this problem on the undersampled dataset.

***

### Best Model & Hyperparameters 🚀

The **XGBoost** model was selected for hyperparameter tuning. The tuning process found the best parameters to be:

`Best Hyperparameters: {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 100}`.

***

### Final Model Performance Visualization 🖼️

The confusion matrix for the optimized XGBoost model on the test data is as follows.



* **True Positives (TP):** 85 fraudulent transactions were correctly identified.
* **False Negatives (FN):** 13 fraudulent transactions were missed. This is the critical error to minimize in fraud detection.
* **True Negatives (TN):** 96 legitimate transactions were correctly identified.
* **False Positives (FP):** 3 legitimate transactions were incorrectly flagged as fraud.

***

### Conclusion ✅

This project successfully demonstrates a complete machine learning pipeline for fraud detection on an imbalanced dataset. It highlights the importance of data preprocessing and the use of appropriate evaluation metrics to build a robust and reliable model. The analysis confirms that ensemble methods like Random Forest and XGBoost are highly effective for this type of classification problem.
