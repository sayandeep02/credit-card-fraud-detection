
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sayandeep02/credit-card-fraud-detection/HEAD)


# Credit Card Fraud Detection : Project Overview
* Handle imbalanced dataset by applying SMOTE and Random Under-Sampling, plus removing the extreme outliers
* Compare which technique fits best to handle the imbalanced datasaet
* Optimized Logistic, K-Nearest Neighbor, Support Vector Machine, and Random Forest Regressors using GridsearchCV to reach the best model.

# **Objective**: 
### In this project you will predict fraudulent credit card transactions with the help of Machine learning models.

![](https://d1m75rqqgidzqn.cloudfront.net/wp-data/2021/09/22153728/iStock-1203763961.jpg)


|ML Models List| Oversampling Techniques Used | Types of Scalers & Transforms Used | Cross Validation Techniques Used
|---|---|---|---|
|Logistic Regression| Random Oversampling | PowerTransformer| StratifiedKFold |
|KNN| SMOTE | RobustScaler | GridSearchCV |
|SVM| ADASYN | - | - |
|Decision Tree| - | - | - |
|Random Forest| - | - | - |
|XGBoost| - | - | - |




# Project Understanding
Suppose you get a call from your bank, and the customer care executive informs you that your card is about to expire in a week. Immediately, you check your card details and realise that it will expire in the next eight days. Now, to renew your membership, the executive asks you to verify a few details such as your credit card number, the expiry date and the CVV number. Will you share these details with the executive?


In such situations, you need to be careful because the details that you might share with them could grant them unhindered access to your credit card account.






Although digital transactions in India registered a 51% growth in 2018–2019, their safety remains a concern. Fraudulent activities have increased severalfold, with approximately 52,304 cases of credit/debit card fraud reported in FY 2019 alone. Owing to this steep increase in banking frauds, it is the need of the hour to detect these fraudulent transactions in time to help consumers and banks that are losing their credit worth each day. Machine learning can play a vital role in detecting fraudulent transactions.


So far, you have learnt about the different types of machine learning models. Now, you will learn which model to choose for your purpose and the reason for it. Understanding models based on different scenarios is an important skill that a data scientist / machine learning engineer should possess. In addition, tuning your model is equally important to get the best fit for your given data.
 
By the end of this module, you will learn how you can build a machine learning model that is capable of detecting fraudulent transactions. You will also learn how to handle class imbalances present in any data set, along with model selection and hyperparameter tuning.
 

 ![](https://dataaspirant.com/wp-content/uploads/2020/09/1-Credit-card-fraud-detection-with-classification-algorithms.png)



 # **Problem Statement**

The problem statement chosen for this project is to predict fraudulent credit card transactions with the help of machine learning models.

In this project, you will analyse customer-level data that has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group. 



# **Business problem overview**
For many banks, retaining high profitable customers is the number one business goal. Banking fraud, however, poses a significant threat to this goal for different banks. In terms of substantial financial losses, trust and credibility, this is a concerning issue to both banks and customers alike.

It has been estimated by Nilson Report that by 2020, banking frauds would account for $30 billion worldwide. With the rise in digital payment channels, the number of fraudulent transactions is also increasing in new and different ways.  

In the banking industry, credit card fraud detection using machine learning is not only a trend but a necessity for them to put proactive monitoring and fraud prevention mechanisms in place. Machine learning is helping these institutions to reduce time-consuming manual reviews, costly chargebacks and fees as well as denials of legitimate transactions.



# **Understanding and defining fraud**
Credit card fraud is any dishonest act or behaviour to obtain information without proper authorisation from the account holder for financial gain. Among different ways of committing frauds, skimming is the most common one, which is a way of duplicating information that is located on the magnetic strip of the card. Apart from this, following are the other ways:

* Manipulation/alteration of genuine cards
* Creation of counterfeit cards
* Stealing/loss of credit cards
* Fraudulent telemarketing




# **About the Dataset**

* The data set is taken from the [Kaggle website](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and has a total of 2,84,807 transactions; out of these, 492 are fraudulent. Since the data set is highly imbalanced, it needs to be handled before model building.

* It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

* The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

* It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

* Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.


# **Metric Used**

##### **ROC-AUC Score**
* The ROC-AUC (Receiver Operating Characteristic - Area Under the Curve) score is a commonly used metric to evaluate the performance of binary classifiers. It measures the ability of a model to distinguish between positive and negative classes by calculating the area under the curve of the receiver operating characteristic curve.

* The ROC-AUC score can be a good metric for imbalanced datasets because it is insensitive to class imbalance. This means that even if the positive and negative classes are not balanced in the dataset, the ROC-AUC score will still provide a reliable measure of the model's ability to discriminate between the two classes.

##### **F1 Score**

* However, it's important to note that the ROC-AUC score does not provide information about the specific performance of the model for each class. For example, if one class is significantly smaller than the other, the model may perform well in the larger class but poorly in the smaller class, and this may not be reflected in the ROC-AUC score. Therefore, it's important to consider other metrics such as precision, recall, F1-score, or confusion matrix when evaluating the performance of a model on imbalanced datasets.

* The F1-score is a commonly used metric for evaluating the performance of binary classifiers. It combines precision and recall into a single score and provides a measure of the overall accuracy of the model.




#### **Precision**

* Precision is a measure of the accuracy of a classification or prediction model. It is defined as the ratio of true positives (correctly predicted positive cases) to the sum of true positives and false positives (incorrectly predicted positive cases) in the model's output. In other words, precision measures the proportion of predicted positive cases that are actually positive.

* A high precision indicates that the model is very good at identifying the positive cases, and there are few false positives in its output. On the other hand, a low precision means that the model has a higher rate of false positives, which can lead to incorrect or misleading results.


#### **Recall**

* Recall is a measure of the completeness of a classification or prediction model. It is defined as the ratio of true positives (correctly predicted positive cases) to the sum of true positives and false negatives (missed positive cases) in the model's output. In other words, recall measures the proportion of actual positive cases that the model correctly identifies.

* A high recall indicates that the model is very good at identifying positive cases, and there are few missed positive cases in its output. On the other hand, a low recall means that the model has a higher rate of missed positive cases, which can also lead to incorrect or misleading results.


# Final Observation on Imbalanced Dataset

#### `A. CROSS VALIDATION -  ROC-AUC Score of the models and best hyperparameters on Imbalanced data`

* **LogisticRegression** {'C': 0.01, 'penalty': 'l2'} = 
  * Best Mean ROC-AUC score for val data: 0.9797969874466093
  * Mean precision val score for best C: 0.885478588591554
  * Mean recall val score for best C: 0.6295975017349064
  * Mean f1 val score for best C: 0.7341406860856002

* **KNeighborsClassifier** {'metric': 'manhattan', 'n_neighbors': 9} = 
  * 0.9274613536399045

* svm.SVC {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 
  * 0.9565173998635063
* **DecisionTreeClassifier** {'criterion': 'entropy', '': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 
  * Best Mean ROC-AUC score for val data: 0.9337472016466822
  * Mean precision val score for best max_depth: 0.8480952241800844
  * Mean recall val score for best max_depth: 0.71578379211967
  * Mean f1 val score for best max_depth: 0.7752315571186218

* RandomForestClassifier {'min_samples_split': 5, 'n_estimators': 500} = 
  * 0.9646808744238831

* **XGBClassifier** {'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.5} = 
  * Best Mean ROC-AUC score for val data: 0.9859738836378374
  * Mean precision val score for best Learning Rate: 0.9233400094242072
  * Mean recall val score for best Learning Rate: 0.779204256303493
  * Mean f1 val score for best Learning Rate: 0.8448234879500908


  #### `B. TEST SET - Metrics & Scores using best model and hyperparameters on Test Set`

* **LogisticRegression** {'C': 0.01, 'penalty': 'l2'} = 
  * LogisticRegression ROC-AUC Score on Test Set = 0.9752271441778737
  * LogisticRegression F1-Score on Test Set = 0.5977011494252873
  * LogisticRegression Precision on Test Set = 0.4785276073619632
  * LogisticRegression Recall on Test Set = 0.7959183673469388 

* **KNeighborsClassifier** {'metric': 'manhattan', 'n_neighbors': 9} = 
  * KNeighbors Classifier ROC-AUC Score on Test Set = 0.9385655570613163
  * KNeighbors Classifier F1-Score on Test Set = 0.824858757062147
  * KNeighbors Classifier Precision on Test Set = 0.9240506329113924
  * KNeighbors Classifier Recall on Test Set = 0.7448979591836735

* **svm.SVC** {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 
  * SVM Classifier ROC-AUC Score on Test Set = 0.9701114653796241
  * SVM Classifier F1 Score on Test Set = 0.8121827411167513
  * SVM Classifier Precision on Test Set = 0.8080808080808081
  * SVM Classifier Recall on Test Set = 0.8163265306122449 

* **DecisionTreeClassifier** {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 
  * **Decision Tree Classifier** ROC-AUC Score on Test Set = 0.9314465304973987
  * Decision Tree Classifier F1-Score on Test Set = 0.8200000000000001
  * Decision Tree Classifier Precision on Test Set = 0.803921568627451
  * Decision Tree Classifier Recall on Test Set = 0.8367346938775511

* **RandomForestClassifier** {'min_samples_split': 5, 'n_estimators': 500} = 
  * Random Forest Classifier ROC-AUC Score on Test Set = 0.9623530686894904
  * Random Forest Classifier F1-Score on Test Set = 0.8282828282828283
  * Random Forest Classifier Precision on Test Set = 0.82
  * Random Forest Classifier Recall on Test Set = 0.8367346938775511

* **XGBClassifier** {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 
  * XGBOOST Classifier ROC-AUC Score on Test Set = 0.9713649035866456
  * XGBOOST Classifier F1-Score on Test Set = 0.7962085308056872
  * XGBOOST Classifier Precision on Test Set = 0.7433628318584071
  * XGBOOST Classifier Recall on Test Set = 0.8571428571428571 

`Best model is LogisticRegression`


### **Table of Scores**

| Model                    | Parameter                                               | ROC-AUC Score | F1-Score   | Precision  | Recall     |
|--------------------------|--------------------------------------------------------|--------------|------------|------------|------------|
| LogisticRegression       | {'C': 0.01, 'penalty': 'l2'}                         | 0.975227144  | 0.59770115 | 0.47852761 | 0.79591836 |
| KNeighborsClassifier     | {'metric': 'manhattan', 'n_neighbors': 9}              | 0.938565557  | 0.82485875 | 0.92405063 | 0.74489795 |
| svm.SVC                  | {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} | 0.970111465  | 0.81218274 | 0.80808081 | 0.81632653 |
| DecisionTreeClassifier   | {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} | 0.931446530  | 0.82000000 | 0.80392156 | 0.83673469 |
| RandomForestClassifier   | {'min_samples_split': 5, 'n_estimators': 500}          | 0.962353068  | 0.82828283 | 0.82000000 | 0.83673469 |
| XGBClassifier            | {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} | 0.971364903  | 0.79620853 | 0.74336283 | 0.85714286 |


1. `Best model is LogisticRegression based on ROC-AUC Score`
2. `Best model is RandomForestClassifier based on F1 Score`



# Model building with balancing Classes

**Why can't we use accuracy for imbalanced dataset?**
* Accuracy is not a good metric for imbalanced datasets.
* This model would receive a very good accuracy score as it predicted correctly for the majority of observations, but this hides the true performance of the model which is objectively not good as it only predicts for one class
* Don't use accuracy score as a metric with imbalanced datasets (will be usually high and misleading), instead use f1-score, precision/recall score or confusion matrix


---


* In **undersampling**, you select fewer data points from the majority class for your model building process to balance both classes.
* In **oversampling**, you assign weights to randomly chosen data points from the minority class. This is done so that the algorithm can focus on this class while optimising the loss function.
* **SMOTE** is a process using which you can generate new data points that lie vectorially between two data points that belong to the minority class.
* **ADASYN** is similar to SMOTE, with a minor change in the sense that the number of synthetic samples that it will add will have a density distribution. The aim here is to create synthetic data for minority examples that are harder to learn rather than the easier ones. 


---


##### **Perform class balancing with** :

I. Random Oversampling

II. SMOTE

III. ADASYN



# I. Random Oversampling

# `A. CROSS VALIDATION -  ROC-AUC Score of the models and best hyperparameters on Imbalanced data`

* **LogisticRegression** {'C': 4, 'penalty': 'l2'} = 
  * Best Mean ROC-AUC score for val data: 0.9884840531068964 `[Before Oversampling {'C': 0.01, 'penalty': 'l2'} = 0.9812052138770543]`
  * Mean precision val score for best C: 0.9719371184117677 `[Before Oversampling {'C': 0.01, 'penalty': 'l2'} = 0.885478588591554]`
  * Mean recall val score for best C: 0.9294177647053652 `[Before Oversampling {'C': 0.01, 'penalty': 'l2'} = 0.6295975017349064]`
  * Mean f1 val score for best C: 0.950201240931848 `[Before Oversampling {'C': 0.01, 'penalty': 'l2'} = 0.7341406860856002]`

* **KNeighborsClassifier** {'n_neighbors': 9} = 
  * 0.9998373276002304 `[Before Oversampling {'metric': 'manhattan', 'n_neighbors': 9} = 0.9274613536399045]`

* **svm.SVC** {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 
  * Not computed due to very large training time `[Before Oversampling {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 0.9565173998635063]`

* **DecisionTreeClassifier** {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 4, 'min_samples_split': 2} = 
  * Best Mean ROC-AUC score for val data: 0.9981460788751075 `[Before Oversampling {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.9337472016466822]`
  * Mean precision val score for best Max Depth: 0.9736125554386047 `[Before Oversampling {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.8480952241800844]`
  * Mean recall val score for best Max Depth: 0.9558410382895657 `[Before Oversampling {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.71578379211967]`
  * Mean f1 val score for best Max Depth: 0.964631942249586 `[Before Oversampling {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.7752315571186218]`

* **RandomForestClassifier** {'min_samples_split': 5, 'n_estimators': 500} = 
  * 1.0 `[Before Oversampling {'min_samples_split': 5, 'n_estimators': 500} = 0.9646808744238831]`

* **XGBClassifier** {'learning_rate': 0.6, 'max_depth': 5, 'subsample': 0.7} =
  * Best Mean ROC-AUC score for val data: 0.9999960678244962 `[Before Oversampling {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.9848866713890976]`
  * Mean precision val score for best Learning Rate: 0.9995517600748944 `[Before Oversampling {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.9233400094242072]`
  * Mean recall val score for best Learning Rate: 1.0 `[Before Oversampling {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.779204256303493]`
  * Mean f1 val score for best Learning Rate: 0.9997758279719619 `[Before Oversampling {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.8448234879500908]`


Proceed with the model which shows the best result
* Apply the best hyperparameter on the model
* Predict on the test dataset

# `B. TEST SET - Metrics & Scores using best model and hyperparameters on Test Set`

* **LogisticRegression** {'C': 4, 'penalty': 'l2'}  = 
  * LogisticRegression ROC-AUC Score on Test Set = 0.9714047244524862 `[LogisticRegression ROC-AUC Score on Test Set {'C': 0.01, 'penalty': 'l2'} = 0.9752271441778737]`
  * LogisticRegression F1-Score on Test Set = 0.932017141909525 `[LogisticRegression F1-Score on Test Set {'C': 0.01, 'penalty': 'l2'} = 0.5977011494252873]`
  * LogisticRegression Precision on Test Set = 0.925193643972344 `[LogisticRegression Precision on Test Set {'C': 0.01, 'penalty': 'l2'} = 0.4785276073619632]`
  * LogisticRegression Recall on Test Set = 0.9389420371412492 `[LogisticRegression Recall on Test Set {'C': 0.01, 'penalty': 'l2'} = 0.7959183673469388 ]`

* **KNeighborsClassifier** {'n_neighbors': 9} = 
  * KNeighbors Classifier ROC-AUC Score on Test Set = 0.9398546705943079 `[KNeighbors Classifier ROC-AUC Score on Test Set {'metric': 'manhattan', 'n_neighbors': 9} = 0.9385655570613163]`
  * KNeighbors Classifier F1-Score on Test Set = 0.9241250283468138 `[KNeighbors Classifier F1-Score on Test Set {'metric': 'manhattan', 'n_neighbors': 9} = 0.824858757062147]`
  * KNeighbors Classifier Precision on Test Set = 0.9986317595164189 `[KNeighbors Classifier Precision on Test Set {'metric': 'manhattan', 'n_neighbors': 9} = 0.9240506329113924]`
  * KNeighbors Classifier Recall on Test Set = 0.8599641249296567 `[KNeighbors Classifier Recall on Test Set {'metric': 'manhattan', 'n_neighbors': 9} = 0.7448979591836735]`

* **svm.SVC** {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 
  * Not computed due to very large training time `[SVM Classifier ROC-AUC Score on Test Set {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 0.9701114653796241]`
  * Not computed due to very large training time `[SVM Classifier F1 Score on Test Set {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 0.8121827411167513]`
  * Not computed due to very large training time `[SVM Classifier Precision on Test Set {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 0.8080808080808081]`
  * Not computed due to very large training time `[SVM Classifier Recall on Test Set {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 0.8163265306122449 ]`

* **DecisionTreeClassifier** {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 4, 'min_samples_split': 2}  = 
  * Decision Tree Classifier ROC-AUC Score on Test Set = 0.9356105342166989 `[Decision Tree Classifier ROC-AUC Score on Test Set {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.9314465304973987]`
  * Decision Tree Classifier F1-Score on Test Set = 0.9189785371841284 `[Decision Tree Classifier F1-Score on Test Set {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.8200000000000001]`
  * Decision Tree Classifier Precision on Test Set = 0.929737415186365 `[Decision Tree Classifier Precision on Test Set {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.803921568627451]`
  * Decision Tree Classifier Recall on Test Set = 0.9084658131682611 `[Decision Tree Classifier Recall on Test Set {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.8367346938775511]`

* **RandomForestClassifier** {'min_samples_split': 5, 'n_estimators': 500} = 
  * Random Forest Classifier ROC-AUC Score on Test Set = 0.9634288542372493 `[Random Forest Classifier ROC-AUC Score on Test Set {'min_samples_split': 5, 'n_estimators': 500} = 0.9623530686894904]`
  * Random Forest Classifier F1-Score on Test Set = 0.8258706467661691 `[Random Forest Classifier F1-Score on Test Set {'min_samples_split': 5, 'n_estimators': 500} = 0.8282828282828283]`
  * Random Forest Classifier Precision on Test Set = 0.8058252427184466 `[Random Forest Classifier Precision on Test Set {'min_samples_split': 5, 'n_estimators': 500} = 0.82]`
  * Random Forest Classifier Recall on Test Set = 0.8469387755102041 `[Random Forest Classifier Recall on Test Set {'min_samples_split': 5, 'n_estimators': 500} = 0.8367346938775511]`

* **XGBClassifier**  {'learning_rate': 0.6, 'max_depth': 5, 'subsample': 0.7} = 
  * XGBOOST Classifier ROC-AUC Score on Test Set = 0.977483446853241 `[XGBOOST Classifier ROC-AUC Score on Test Set {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.9713649035866456]`
  * XGBOOST Classifier F1-Score on Test Set = 0.9368538930690137 `[XGBOOST Classifier F1-Score on Test Set {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.7962085308056872]`
  * XGBOOST Classifier Precision on Test Set = 0.9888057514603317 `[XGBOOST Classifier Precision on Test Set {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.7433628318584071]`
  * XGBOOST Classifier Recall on Test Set = 0.8900886325267304 `[XGBOOST Classifier Recall on Test Set {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.8571428571428571 ]`

---


  > * `Based on ROC-AUC Scores-`
    *  KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier & XGBClassifier models scores increase after Oversampling. 
    * LogisticRegression score decreased. But, the best model is XGBClassifier 

  > * `Based on F1 Scores-`
    *  LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier & XGBClassifier models scores increase after Oversampling. 
    *  RandomForestClassifier score decreased. But, the best model is XGBClassifier 

---


| Model | Parameter | ROC-AUC Score | F1-Score | Precision |	Recall |
|---|---|---|---|---|---|
| LogisticRegression | {'C': 4, 'penalty': 'l2'} | 0.9714047245 | 0.9320171419 | 0.925193644 | 0.9389420371 |
| KNeighborsClassifier |	{'n_neighbors': 9} | 0.9398546706 | 0.9241250283 | 0.9986317595 | 0.8599641249 |
| svm.SVC | {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} | NA | NA | NA | NA |
| DecisionTreeClassifier | {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 4, 'min_samples_split': 2} | 0.9356105342 | 0.9189785372 | 0.9297374152 | 0.9084658132 |
| RandomForestClassifier | {'min_samples_split': 5, 'n_estimators': 500} | 0.9634288542 | 0.8258706468 | 0.8058252427 | 0.8469387755 |
| XGBClassifier | {'learning_rate': 0.6, 'max_depth': 5, 'subsample': 0.7} | 0.9774834469 | 0.9368538931 | 0.9888057515 | 0.8900886325 |



# **`II. SMOTE`**
Synthetic Minority Over-sampling Technique
- Build different models on the balanced dataset and see the result



# `A. CROSS VALIDATION -  ROC-AUC Score of the models and best hyperparameters on Imbalanced data`

* **LogisticRegression** {'C': 4, 'penalty': 'l2'} = 
  * Best Mean ROC-AUC score for val data: 0.99074791351665 `[Before Oversampling {'C': 0.01, 'penalty': 'l2'} = 0.9812052138770543]`
  * Mean precision val score for best C: 0.9700892879660502 `[Before Oversampling {'C': 0.01, 'penalty': 'l2'} = 0.885478588591554]`
  * Mean recall val score for best C: 0.9292199198948344 `[Before Oversampling {'C': 0.01, 'penalty': 'l2'} = 0.6295975017349064]`
  * Mean f1 val score for best C: 0.9492131800549414 `[Before Oversampling {'C': 0.01, 'penalty': 'l2'} = 0.7341406860856002]`

* **KNeighborsClassifier** {'n_neighbors': 9} = 
  * 0.9998373276002304 `[Before Oversampling {'metric': 'manhattan', 'n_neighbors': 9} = 0.9274613536399045]`

* **svm.SVC** {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 
  * Not computed due to very large training time `[Before Oversampling {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 0.9565173998635063]`

* **DecisionTreeClassifier** {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 4, 'min_samples_split': 2} = 
  * Best Mean ROC-AUC score for val data: 0.9947661967283646 `[Before Oversampling {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.9337472016466822]`
  * Mean precision val score for best Max Depth: 0.9559803470320745 `[Before Oversampling {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.8480952241800844]`
  * Mean recall val score for best Max Depth: 0.956135607229689 `[Before Oversampling {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.71578379211967]`
  * Mean f1 val score for best Max Depth: 0.9560475733357633 `[Before Oversampling {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.7752315571186218]`

* **RandomForestClassifier** {'min_samples_split': 5, 'n_estimators': 500} = 
  * 1.0 `[Before Oversampling {'min_samples_split': 5, 'n_estimators': 500} = 0.9646808744238831]`

* **XGBClassifier** {'learning_rate': 0.8, 'max_depth': 5, 'subsample': 0.9} =
  * Best Mean ROC-AUC score for val data: 0.9999932782930695
  `[Before Oversampling {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.9848866713890976]`
  * Mean precision val score for best Learning Rate: 0.9985994214344757 `[Before Oversampling {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.9233400094242072]`
  * Mean recall val score for best Learning Rate: 0.9999648275892389 `[Before Oversampling {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.779204256303493]`
  * Mean f1 val score for best Learning Rate: 0.9992816564080629 `[Before Oversampling {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.8448234879500908]`

# `B. TEST SET - Metrics & Scores using best model and hyperparameters on Test Set`
* **LogisticRegression** {'C': 4, 'penalty': 'l2'}  = 
  * LogisticRegression ROC-AUC Score on Test Set = 0.969831420233101 `[LogisticRegression ROC-AUC Score on Test Set {'C': 0.01, 'penalty': 'l2'} = 0.9752271441778737]`
  * LogisticRegression F1-Score on Test Set = 0.9210604136587401 `[LogisticRegression F1-Score on Test Set {'C': 0.01, 'penalty': 'l2'} = 0.5977011494252873]`
  * LogisticRegression Precision on Test Set = 0.9111856823266219 `[LogisticRegression Precision on Test Set {'C': 0.01, 'penalty': 'l2'} = 0.4785276073619632]`
  * LogisticRegression Recall on Test Set = 0.9311515194147439 `[LogisticRegression Recall on Test Set {'C': 0.01, 'penalty': 'l2'} = 0.7959183673469388 ]`

* **KNeighborsClassifier** {'n_neighbors': 9} = 
  * KNeighbors Classifier ROC-AUC Score on Test Set = 0.9520626163291522 `[KNeighbors Classifier ROC-AUC Score on Test Set {'metric': 'manhattan', 'n_neighbors': 9} = 0.9385655570613163]`
  * KNeighbors Classifier F1-Score on Test Set = 0.9379643836890316 `[KNeighbors Classifier F1-Score on Test Set {'metric': 'manhattan', 'n_neighbors': 9} = 0.824858757062147]`
  * KNeighbors Classifier Precision on Test Set = 0.995283298138975 `[KNeighbors Classifier Precision on Test Set {'metric': 'manhattan', 'n_neighbors': 9} = 0.9240506329113924]`
  * KNeighbors Classifier Recall on Test Set = 0.8868880135059088 `[KNeighbors Classifier Recall on Test Set {'metric': 'manhattan', 'n_neighbors': 9} = 0.7448979591836735]`

* **svm.SVC** {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 
  * Not computed due to very large training time `[SVM Classifier ROC-AUC Score on Test Set {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 0.9701114653796241]`
  * Not computed due to very large training time `[SVM Classifier F1 Score on Test Set {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 0.8121827411167513]`
  * Not computed due to very large training time `[SVM Classifier Precision on Test Set {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 0.8080808080808081]`
  * Not computed due to very large training time `[SVM Classifier Recall on Test Set {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 0.8163265306122449 ]`

* **DecisionTreeClassifier** {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2}  = 
  * Decision Tree Classifier ROC-AUC Score on Test Set = 0.9484931498819542 `[Decision Tree Classifier ROC-AUC Score on Test Set {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.9314465304973987]`
  * Decision Tree Classifier F1-Score on Test Set = 0.9285663540778629 `[Decision Tree Classifier F1-Score on Test Set {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.8200000000000001]`
  * Decision Tree Classifier Precision on Test Set = 0.9378946613088404 `[Decision Tree Classifier Precision on Test Set {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.803921568627451]`
  * Decision Tree Classifier Recall on Test Set = 0.9194217782779966 `[Decision Tree Classifier Recall on Test Set {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.8367346938775511]`

* **RandomForestClassifier** {'min_samples_split': 5, 'n_estimators': 500} = 
  * Random Forest Classifier ROC-AUC Score on Test Set = 0.9788061953689164 `[Random Forest Classifier ROC-AUC Score on Test Set {'min_samples_split': 5, 'n_estimators': 500} = 0.9623530686894904]`
  * Random Forest Classifier F1-Score on Test Set = 0.8770312160641301 `[Random Forest Classifier F1-Score on Test Set {'min_samples_split': 5, 'n_estimators': 500} = 0.8282828282828283]`
  * Random Forest Classifier Precision on Test Set = 0.9997524195363493 `[Random Forest Classifier Precision on Test Set {'min_samples_split': 5, 'n_estimators': 500} = 0.82]`
  * Random Forest Classifier Recall on Test Set = 0.7811444850872257 `[Random Forest Classifier Recall on Test Set {'min_samples_split': 5, 'n_estimators': 500} = 0.8367346938775511]`

* **XGBClassifier**  {'learning_rate': 0.8, 'max_depth': 5, 'subsample': 0.9} = 
  * XGBOOST Classifier ROC-AUC Score on Test Set = 0.9921570289753232 `[XGBOOST Classifier ROC-AUC Score on Test Set {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.9713649035866456]`
  * XGBOOST Classifier F1-Score on Test Set = 0.9248408605806464 `[XGBOOST Classifier F1-Score on Test Set {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.7962085308056872]`
  * XGBOOST Classifier Precision on Test Set = 0.9988575625280509 `[XGBOOST Classifier Precision on Test Set {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.7433628318584071]`
  * XGBOOST Classifier Recall on Test Set = 0.8610368598761958 `[XGBOOST Classifier Recall on Test Set {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.8571428571428571 ]`

---


  > * `Based on ROC-AUC Scores-`
    *  KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier & XGBClassifier models scores increase after Oversampling. 
    * LogisticRegression score decreased. But, the best model is XGBClassifier 

  > * `Based on F1 Scores-`
    *  LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier & XGBClassifier models scores increase after Oversampling. 
    *  RandomForestClassifier score decreased. But, the best model is XGBClassifier 

---



| Model | Parameter | ROC-AUC Score | F1-Score | Precision | Recall |
|---|---|---|---|---|---|
| LogisticRegression | {'C': 4, 'penalty': 'l2'}  | 0.9698314202 | 0.9210604137 | 0.9111856823 | 0.9311515194 |
| KNeighborsClassifier |	{'metric': 'manhattan', 'n_neighbors': 9} |	0.9520626163 | 0.9379643837 | 0.9952832981 | 0.8868880135 |
| svm.SVC |	{'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} | NA | NA | NA | NA |
| DecisionTreeClassifier |	{'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} | 0.9484931499 | 0.9285663541 | 0.9378946613 | 0.9194217783 |
| RandomForestClassifier |	{'min_samples_split': 5, 'n_estimators': 500} | 0.9788061954 | 0.8770312161 | 0.9997524195 | 0.7811444851 |
| XGBClassifier | {'learning_rate': 0.8, 'max_depth': 5, 'subsample': 0.9} | 0.992157029 | 0.9248408606 | 0.9988575625 | 0.8610368599 |


# **`III. ADASYN`**



# `A. CROSS VALIDATION -  ROC-AUC Score of the models and best hyperparameters on Imbalanced data`

* **LogisticRegression** {'C': 4, 'penalty': 'l2'} = 
  * Best Mean ROC-AUC score for val data: 0.9600512812300878 `[Before Oversampling {'C': 0.01, 'penalty': 'l2'} = 0.9812052138770543]`
  * Mean precision val score for best C: 0.9103281184583857 `[Before Oversampling {'C': 0.01, 'penalty': 'l2'} = 0.885478588591554]`
  * Mean recall val score for best C: 0.8684424127160906 `[Before Oversampling {'C': 0.01, 'penalty': 'l2'} = 0.6295975017349064]`
  * Mean f1 val score for best C: 0.8884981922559546 `[Before Oversampling {'C': 0.01, 'penalty': 'l2'} = 0.7341406860856002]`

* **KNeighborsClassifier** {'n_neighbors': 9} = 
  * 0.9860334294397143 `[Before Oversampling {'metric': 'manhattan', 'n_neighbors': 9} = 0.9274613536399045]`

* **svm.SVC** {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 
  * Not computed due to very large training time `[Before Oversampling {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 0.9565173998635063]`

* **DecisionTreeClassifier** {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 4, 'min_samples_split': 2} = 
  * Best Mean ROC-AUC score for val data: 0.9387163825214735 `[Before Oversampling {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.9337472016466822]`
  * Mean precision val score for best Max Depth: 0.908528931388625 `[Before Oversampling {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.8480952241800844]`
  * Mean recall val score for best Max Depth: 0.8673782646963896 `[Before Oversampling {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.71578379211967]`
  * Mean f1 val score for best Max Depth: 0.8871749175083493 `[Before Oversampling {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.7752315571186218]`

* **RandomForestClassifier** {'min_samples_split': 5, 'n_estimators': 500} = 
  * 0.9961281862850145 `[Before Oversampling {'min_samples_split': 5, 'n_estimators': 500} = 0.9646808744238831]`

* **XGBClassifier** {'learning_rate': 0.8, 'max_depth': 5, 'subsample': 0.9} =
  * Best Mean ROC-AUC score for val data: 0.9979584458494518 
  `[Before Oversampling {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.9848866713890976]`
  * Mean precision val score for best Learning Rate: 0.9984003044486842 `[Before Oversampling {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.9233400094242072]`
  * Mean recall val score for best Learning Rate: 0.8774460593069945 `[Before Oversampling {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.779204256303493]`
  * Mean f1 val score for best Learning Rate: 0.933889867542906 `[Before Oversampling {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.8448234879500908]`

# `B. TEST SET - Metrics & Scores using best model and hyperparameters on Test Set`
* **LogisticRegression** {'C': 4, 'penalty': 'l2'}  = 
  * LogisticRegression ROC-AUC Score on Test Set = 0.9173743861170938 `[LogisticRegression ROC-AUC Score on Test Set {'C': 0.01, 'penalty': 'l2'} = 0.9752271441778737]`
  * LogisticRegression F1-Score on Test Set = 0.8375936925470231 `[LogisticRegression F1-Score on Test Set {'C': 0.01, 'penalty': 'l2'} = 0.5977011494252873]`
  * LogisticRegression Precision on Test Set = 0.8420444649807176 `[LogisticRegression Precision on Test Set {'C': 0.01, 'penalty': 'l2'} = 0.4785276073619632]`
  * LogisticRegression Recall on Test Set = 0.8331897233896636 `[LogisticRegression Recall on Test Set {'C': 0.01, 'penalty': 'l2'} = 0.7959183673469388 ]`

* **KNeighborsClassifier** {'n_neighbors': 9} = 
  * KNeighbors Classifier ROC-AUC Score on Test Set = 0.8685620474989183 `[KNeighbors Classifier ROC-AUC Score on Test Set {'metric': 'manhattan', 'n_neighbors': 9} = 0.9385655570613163]`
  * KNeighbors Classifier F1-Score on Test Set = 0.823434809950413 `[KNeighbors Classifier F1-Score on Test Set {'metric': 'manhattan', 'n_neighbors': 9} = 0.824858757062147]`
  * KNeighbors Classifier Precision on Test Set = 0.9950914436637265 `[KNeighbors Classifier Precision on Test Set {'metric': 'manhattan', 'n_neighbors': 9} = 0.9240506329113924]`
  * KNeighbors Classifier Recall on Test Set = 0.7022877943271142 `[KNeighbors Classifier Recall on Test Set {'metric': 'manhattan', 'n_neighbors': 9} = 0.7448979591836735]`

* **svm.SVC** {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 
  * Not computed due to very large training time `[SVM Classifier ROC-AUC Score on Test Set {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 0.9701114653796241]`
  * Not computed due to very large training time `[SVM Classifier F1 Score on Test Set {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 0.8121827411167513]`
  * Not computed due to very large training time `[SVM Classifier Precision on Test Set {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 0.8080808080808081]`
  * Not computed due to very large training time `[SVM Classifier Recall on Test Set {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} = 0.8163265306122449 ]`

* **DecisionTreeClassifier** {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 4, 'min_samples_split': 2}  = 
  * Decision Tree Classifier ROC-AUC Score on Test Set = 0.8812439127343993 `[Decision Tree Classifier ROC-AUC Score on Test Set {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.9314465304973987]`
  * Decision Tree Classifier F1-Score on Test Set = 0.8003628524887584 `[Decision Tree Classifier F1-Score on Test Set {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.8200000000000001]`
  * Decision Tree Classifier Precision on Test Set = 0.8986507206378411 `[Decision Tree Classifier Precision on Test Set {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.803921568627451]`
  * Decision Tree Classifier Recall on Test Set = 0.721455325584258 `[Decision Tree Classifier Recall on Test Set {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} = 0.8367346938775511]`

* **RandomForestClassifier** {'min_samples_split': 5, 'n_estimators': 500} = 
  * Random Forest Classifier ROC-AUC Score on Test Set = 0.9634288542372493 `[Random Forest Classifier ROC-AUC Score on Test Set {'min_samples_split': 5, 'n_estimators': 500} = 0.9623530686894904]`
  * Random Forest Classifier F1-Score on Test Set = 0.8258706467661691 `[Random Forest Classifier F1-Score on Test Set {'min_samples_split': 5, 'n_estimators': 500} = 0.8282828282828283]`
  * Random Forest Classifier Precision on Test Set = 0.8058252427184466 `[Random Forest Classifier Precision on Test Set {'min_samples_split': 5, 'n_estimators': 500} = 0.82]`
  * Random Forest Classifier Recall on Test Set = 0.8469387755102041 `[Random Forest Classifier Recall on Test Set {'min_samples_split': 5, 'n_estimators': 500} = 0.8367346938775511]`

* **XGBClassifier**  {'learning_rate': 0.8, 'max_depth': 5, 'subsample': 0.9} = 
  * XGBOOST Classifier ROC-AUC Score on Test Set = 0.9679589101349129 `[XGBOOST Classifier ROC-AUC Score on Test Set {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.9713649035866456]`
  * XGBOOST Classifier F1-Score on Test Set = 0.7803322202243343 `[XGBOOST Classifier F1-Score on Test Set {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.7962085308056872]`
  * XGBOOST Classifier Precision on Test Set = 0.9969643101326405 `[XGBOOST Classifier Precision on Test Set {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.7433628318584071]`
  * XGBOOST Classifier Recall on Test Set = 0.6410396187595618 `[XGBOOST Classifier Recall on Test Set {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5} = 0.8571428571428571 ]`

---


  > * `Based on ROC-AUC Scores-`
    *  KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier & XGBClassifier models scores increase after Oversampling. 
    * LogisticRegression score decreased. But, the best model is XGBClassifier 

  > * `Based on F1 Scores-`
    *  LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier & XGBClassifier models scores increase after Oversampling. 
    *  RandomForestClassifier score decreased. But, the best model is XGBClassifier 

---


| Model	| Parameter	| ROC-AUC Score	| F1-Score	| Precision	| Recall |
|---|---|---|---|---|---|
| LogisticRegression | {'C': 4, 'penalty': 'l2'}  |	0.9173743861 | 0.8375936925 | 0.842044465 | 0.8331897234 |
| KNeighborsClassifier | {'n_neighbors': 9} | 0.8685620475 | 0.82343481 | 0.9950914437	| 0.7022877943 |
| svm.SVC	| {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} | NA | NA | NA | NA |
| DecisionTreeClassifier | {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 4, 'min_samples_split': 2} | 0.8812439127 | 0.8003628525	| 0.8986507206	| 0.7214553256 |
| RandomForestClassifier | {'min_samples_split': 5, 'n_estimators': 500} | 0.9634288542 | 0.8258706468 | 0.8058252427	| 0.8469387755 |
| XGBClassifier | {'learning_rate': 0.8, 'max_depth': 5, 'subsample': 0.9} | 0.9679589101 | 0.7803322202	| 0.9969643101	| 0.6410396188 |

    1. Model sensitivity: Some models may be more sensitive to the effects of oversampling than others. 
    For example, decision trees may be less affected by oversampling than linear models.
    2. Overfitting: The oversampled data may introduce too much noise and overfit the model to the training data. 
    This can result in poor generalization performance on new, unseen data.

# Model For each Type of OverSampling

# 1. LogisticRegression

| Type of OverSampling | Model                | Parameter                        | ROC-AUC Score | F1-Score     | Precision    | Recall       |
|----------------------|----------------------|----------------------------------|---------------|--------------|--------------|--------------|
| None                 | LogisticRegression | {'C': 0.01, 'penalty': 'l2'}     | 0.9752271442  | 0.5977011494 | 0.4785276074 | 0.7959183673 |
| ROS                  | LogisticRegression | {'C': 4, 'penalty': 'l2'}        | 0.9714047245  | 0.9320171419 | 0.925193644  | 0.9389420371 |
| SMOTE                | LogisticRegression | {'C': 4, 'penalty': 'l2'}        | 0.9698314202  | 0.9210604137 | 0.9111856823 | 0.9311515194 |
| ADASYN               | LogisticRegression | {'C': 4, 'penalty': 'l2'}        | 0.9173743861  | 0.8375936925 | 0.842044465  | 0.8331897234 |

# 2. KNeighborsClassifier

| Type of OverSampling | Model                  | Parameter                                      | ROC-AUC Score | F1-Score      | Precision     | Recall        |
|----------------------|------------------------|------------------------------------------------|---------------|---------------|---------------|---------------|
| None                 | KNeighborsClassifier | {'metric': 'manhattan', 'n_neighbors': 9}       | 0.9385655571  | 0.8248587571  | 0.9240506329  | 0.7448979592  |
| ROS                  | KNeighborsClassifier | {'n_neighbors': 9}                             | 0.9398546706  | 0.9241250283  | 0.9986317595  | 0.8599641249  |
| SMOTE                | KNeighborsClassifier | {'metric': 'manhattan', 'n_neighbors': 9}                             | 0.9520626163  | 0.9379643837  | 0.9952832981  | 0.8868880135  |
| ADASYN               | KNeighborsClassifier | {'n_neighbors': 9}                             | 0.8685620475  | 0.82343481    | 0.9950914437  | 0.7022877943  |


# 3. svm.SVC
| Type of OverSampling | Model   | Parameter                                                       | ROC-AUC Score | F1-Score | Precision | Recall |
|----------------------|---------|-----------------------------------------------------------------|---------------|---------------|---------------|---------------|
| None                 | svm.SVC | {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} | 0.9701114654  | 0.8121827411  | 0.8080808081  | 0.8163265306  |
| ROS                  | svm.SVC | {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} | NA            | NA            | NA            | NA            |
| SMOTE                | svm.SVC | {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} | NA            | NA            | NA            | NA            |
| ADASYN               | svm.SVC | {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'probability': True} | NA            | NA            | NA            | NA            |



# 4. DecisionTreeClassifier

| Type of OverSampling | Model                     | Parameter                                                                                       | ROC-AUC Score | F1-Score     | Precision    | Recall       |
|----------------------|---------------------------|------------------------------------------------------------------------------------------------|---------------|--------------|--------------|--------------|
| None                 | DecisionTreeClassifier    | {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2}         | 0.9484931499  | 0.9285663541 | 0.9378946613 | 0.9194217783 |
| ROS                  | DecisionTreeClassifier    | {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 4, 'min_samples_split': 2}         | 0.9356105342  | 0.9189785372 | 0.9297374152 | 0.9084658132 |
| SMOTE                | DecisionTreeClassifier    | {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2}         | 0.9484931499  | 0.9285663541 | 0.9378946613 | 0.9194217783 |
| ADASYN               | DecisionTreeClassifier    | {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 4, 'min_samples_split': 2}         | 0.8812439127  | 0.8003628525 | 0.8986507206 | 0.7214553256 |


# 5. RandomForestClassifier

| Type of OverSampling | Model                   | Parameter                                        | ROC-AUC Score | F1-Score     | Precision   | Recall       |
|----------------------|-------------------------|-------------------------------------------------|---------------|--------------|-------------|--------------|
| None                 | RandomForestClassifier | {'min_samples_split': 5, 'n_estimators': 500}     | 0.9623530686894904 | 0.8282828283 | 0.82        | 0.8367346939 |
| ROS                  | RandomForestClassifier | {'min_samples_split': 5, 'n_estimators': 500}     | 0.9634288542372493 | 0.8258706468 | 0.8058252427| 0.8469387755 |
| SMOTE                | RandomForestClassifier | {'min_samples_split': 5, 'n_estimators': 500}     | 0.9788061953689164| 0.8258706468 | 0.8058252427| 0.8469387755 |
| ADASYN               | RandomForestClassifier | {'min_samples_split': 5, 'n_estimators': 500}     | 0.9634288542372493  | 0.8258706468 | 0.8058252427| 0.8469387755 |


# 6. XGBClassifier

| Type of OverSampling | Model                 | Parameter                                                           | ROC-AUC Score | F1-Score      | Precision     | Recall        |
|----------------------|-----------------------|---------------------------------------------------------------------|---------------|---------------|---------------|---------------|
| None                 | XGBClassifier         | {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5}             | 0.9713649036  | 0.7962085308  | 0.7433628319  | 0.8571428571  |
| ROS                  | XGBClassifier         | {'learning_rate': 0.6, 'max_depth': 5, 'subsample': 0.7}             | 0.9774834469  | 0.9368538931  | 0.9888057515  | 0.8900886325  |
| SMOTE                | XGBClassifier         | {'learning_rate': 0.8, 'max_depth': 5, 'subsample': 0.9}             | 0.992157029   | 0.9248408606  | 0.9988575625  | 0.8610368599  |
| ADASYN               | XGBClassifier         | {'learning_rate': 0.8, 'max_depth': 5, 'subsample': 0.9}             | 0.9679589101  | 0.7803322202  | 0.9969643101  | 0.6410396188  |




# Best Oversampling Technique and Hyper Parameter for Each model

* Best LogisticRegression Model Performance: 0.9752271442 {'C': 0.01, 'penalty': 'l2'} `[None]`
* Best KNeighborsClassifier Model Performance: 0.9520626163 {'metric': 'manhattan', 'n_neighbors': 9} `[SMOTE]`
* Best svm.SVC Model Performance: NA [Due to High Training Time]
* Best DecisionTreeClassifier Model Performance: 0.9484931499 {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} `[SMOTE]` 
* Best RandomForestClassifier Model Performance: 0.9788061953689164 {'min_samples_split': 5, 'n_estimators': 500} `[SMOTE]`
* Best XGBClassifier Model Performance: 0.9921570289753232 {'learning_rate': 0.8, 'max_depth': 5, 'subsample': 0.9} `[SMOTE]`


# **Best Model is XGBOOST CLassifier with SMOTE:** `0.992157029`


## choosing the best threshold
The Threshold of `0.9880` gives a Precision of `99.42%` and `F1 Score` of `99.56%` on Training data. 


