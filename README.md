Smoking Cessation Detection Using Biosignals
============================================

Project Overview
----------------

This project investigates and compares various machine learning methodologies to detect smoking status using biosignal-based health indicators, not just centring solely around binary classification. It explores the effect of preprocessing techniques, dimensionality reduction, clustering, and multiple classification algorithms—individually and in ensembles—on model performance.

Dataset Description
-------------------

**Source:** [Kaggle - Smoker Status Prediction Using Biosignals](https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction-using-biosignals)

The dataset includes a wide range of physiological and medical features with a binary target variable smoking (1: smoker, 0: non-smoker). The features span:

*   **Demographics**: Age, height, weight, waist circumference
    
*   **Sensory Metrics**: Eyesight (left/right), hearing (left/right)
    
*   **Vital Signs**: Systolic and diastolic blood pressure
    
*   **Biochemical Measurements**: Cholesterol, HDL, LDL, triglyceride, fasting blood sugar
    
*   **Liver and Kidney Function**: AST, ALT, GTP, serum creatinine
    
*   **Other Health Indicators**: Hemoglobin, dental caries, urine protein
    

The dataset is clean with no missing values and includes both categorical and continuous variables.

Methodology Overview
--------------------

### 1\. Exploratory Data Analysis (EDA)

*   Examined class distribution
    
*   Analyzed null values, datatypes, and duplicates
    
*   Used histograms and violin plots to visualize feature distributions
    
*   Derived insights such as elevated lipid profiles and liver enzymes among smokers
    

### 2\. Data Preprocessing

*   Outlier handling (domain-based, IQR, and range-based capping)
    
*   Encoding of categorical features
    
*   Feature reduction via low-variance filtering and correlation analysis
    
*   Standardization of features
    

### 3\. Dimensionality Reduction

*   **Linear Discriminant Analysis (LDA)**: Used to visualize class separation
    
*   **Principal Component Analysis (PCA)**: Retained ~97.5% of data variance
    

### 4\. Unsupervised Clustering

*   KMeans clustering applied post-PCA
    
*   Validated using Elbow Method, Silhouette Score, and Davies-Bouldin Index
    
*   Cluster labels appended as new features to enhance classification
    

### 5\. Classification Models

Trained and compared multiple classifiers on PCA-only and PCA+Clustered datasets:

*   K-Nearest Neighbors (KNN)
    
*   Gaussian Naive Bayes (GNB)
    
*   Decision Tree (tuned with GridSearchCV)
    

### 6\. Ensemble Learning

Tested multiple ensemble strategies:

*   Hard Voting
    
*   Soft Voting
    
*   Weighted Voting (final selected model)
    

Results Summary
---------------

### Comparison of Individual Models and Ensembles


| Model                          | ROC-AUC | Accuracy | Precision | Recall | F1 Score |
|-------------------------------|---------|----------|-----------|--------|----------|
| KNN (PCA Only)                | 0.83    | 0.75     | 0.69      | 0.78   | 0.73     |
| Naive Bayes (PCA + Clustered) | 0.79    | 0.70     | 0.61      | 0.90   | 0.73     |
| Decision Tree (PCA Only)      | 0.81    | 0.73     | 0.67      | 0.78   | 0.72     |
| Soft Voting Ensemble          | 0.83    | 0.73     | 0.63      | 0.90   | 0.74     |
| Weighted Voting Ensemble      | 0.83    | 0.74     | 0.65      | 0.87   | 0.75     |


### Model Comparison Findings

*   **Weighted Voting Ensemble** delivered the highest ROC-AUC (0.8305), along with the best accuracy and F1 Score.
    
*   **KNN (PCA Only)** was the top-performing individual model based on ROC-AUC.
    
*   **Naive Bayes** achieved the highest recall (0.9004), indicating strong performance in identifying smokers.
    
*   **Soft Voting** emphasised recall, while **Weighted Voting** maintained a better balance between precision and recall.
    

### Conclusion

The **Weighted Voting Ensemble** emerged as the most robust model:

*   Best ROC-AUC and F1 Score
    
*   Balanced performance across precision and recall
    
*   Strong candidate for real-world health risk prediction systems
    

How to Run
----------

1.  Install dependencies:

`   pip install -r requirements.txt   `

1.  Download the dataset and place it in the data/ folder
    
2.  Open and execute Project.ipynb using Jupyter
    

References
----------

*   Dataset: [Kaggle](https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction-using-biosignals)
    
*   Scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org/)
    
*   Voting Classifier: [https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier)
    

Acknowledgments
---------------

Thanks to the dataset provider and the open-source ML ecosystem that made this comparative analysis possible.
