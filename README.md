# Deep-Learning-Pima-Diabetes
Binary classification model for diabetes prediction using PyTorch. Features 5-fold cross-validation, early stopping, and L2 regularization, achieving 75% accuracy on the Pima Indians Diabetes dataset.


# Overview of the Project:
This repository contains the work submitted for the Deep Learning assignment. It implements a binary classification model using PyTorch to predict diabetes from the Pima Indians Diabetes dataset. Some of the key features of the project are 5-fold cross-validation, early stopping (patience=5), L2 regularization (weight_decay=0.01), and model evaluation with metrics such as Accuracy, Precision, Recall, F1-Score, and AUC.

-Dataset: Pima Indians Diabetes Database from Kaggle, containing 768 samples and 8 features.
-Model: Custom feedforward neural network, 3 layers, 8-64-32-1, ReLU/Sigmoid.
-Improvements: Cross-validation for robustness, early stopping in order to avoid overfitting and L2 regularization for weight penalty.
-Results: Avg Accuracy 0.7500, AUC 0.8410 (from 5-fold CV).

# Overview of Files in This Repository:
Jupyter Notebook containing complete code (data loading, pre-processing, model, training and evaluation).
Detailed report (comparison, fitting curves, metrics).
-requirements.txt: List of Python dependencies.
-README: This file.
-.gitignore: Ignores unnecessary files.
-LICENSE: License under which the project is being made available.

# How to Run the Code
1.Prerequisites:
Install Python 3.8+ and Jupyter Notebook.
- Download the dataset: diabetes.csv from Kaggle at https://www.kaggle.com/uciml/pima-indians-diabetes-database and store it in the project folder.
  
2.Install Dependencies:
- Run: pip install -r requirements.txt

3.Run the Notebook:
- Open ipynb file in Jupyter.
- Execute all cells in order (Cells 1-5). 5-fold CV may take 5-10 minutes.
- Expected Output: Fold-by-fold training logs, metrics, loss curves plot, and averaged results, such as Avg Accuracy: 0.7500.

4.Troubleshooting:
- If "File not found" for CSV then ensure the file is in the folder.

If PyTorch fails: Check GPU/CPU compatibility.
- For custom epochs/loss checks: Modify the code according to the comments available in the notebook.
  
# Key Outputs
- **Metrics**: Accuracy 0.7500, Precision 0.6774, Recall 0.5403, F1 0.5996, AUC 0.8410.
- **Loss Curves**: Plotted in the notebook: average train/val loss across folds.
- **Comparisons**: Benchmarked against Logistic Regression (0.7722), Random Forest (0.7579), SVM (0.7514), and XGBoost (state-of-the-art: ~0.7800).

  # Acknowledgement
- Data from Kaggle.
- PyTorch for deep learning framework.
- Sklearn for preprocessing and comparisons.
