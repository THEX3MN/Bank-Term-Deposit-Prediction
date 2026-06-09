# Bank Term Deposit Prediction using Machine Learning

A machine learning project developed as part of an academic assignment to predict whether a bank customer will subscribe to a term deposit based on customer details, campaign information, and economic indicators.

The project uses the Bank Marketing dataset and implements two supervised machine learning models: Decision Tree and k-Nearest Neighbours. It includes data cleaning, categorical encoding, feature scaling for k-NN, train-test splitting, model training, performance evaluation, and visualisations.

## Project Overview

A bank conducted direct marketing campaigns through phone calls to promote term deposit subscriptions. The goal of this project is to build machine learning models that can predict whether a client will subscribe to a term deposit based on available customer and campaign attributes.

The target variable is:

* `y = yes` if the client subscribed
* `y = no` if the client did not subscribe

## Dataset

The dataset includes customer, campaign, and economic attributes such as:

* Age
* Job
* Marital status
* Education
* Default status
* Housing loan
* Personal loan
* Contact type
* Month and day of contact
* Call duration
* Campaign contacts
* Previous campaign outcome
* Employment variation rate
* Consumer price index
* Consumer confidence index
* Euribor 3-month rate
* Number of employees
* Target variable: `y`

## Machine Learning Models Used

### Decision Tree Classifier

The Decision Tree model was used to classify whether a customer would subscribe to a term deposit. It was evaluated using accuracy, precision, recall, F1 score, confusion matrix, ROC curve, AUC score, feature importance, and a visualised tree structure.

### k-Nearest Neighbours

The k-NN model was used as a distance-based classification algorithm. Min-Max scaling was applied before training because k-NN is sensitive to feature scale. The model was evaluated using accuracy, precision, recall, F1 score, confusion matrix, feature space visualisation, and accuracy comparison across different k values.

## Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn
* Decision Tree Classifier
* k-Nearest Neighbours
* Min-Max Scaling
* Data preprocessing
* Model evaluation
* Data visualisation

## Project Workflow

1. Loaded and inspected the bank marketing dataset
2. Checked for missing values and duplicate records
3. Removed duplicate and missing records
4. Encoded categorical variables into numerical values
5. Separated features and target variable
6. Applied feature normalisation for the k-NN model
7. Split the dataset into training and testing sets
8. Trained Decision Tree and k-NN classification models
9. Generated predictions using the test dataset
10. Evaluated model performance using classification metrics
11. Produced visualisations including ROC curve, feature importance, feature space, and accuracy vs k

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC Curve
* AUC Score

## Visualisations

The project includes visualisations such as:

* Decision Tree ROC Curve
* Decision Tree Feature Importance
* Decision Tree Structure
* k-NN Feature Space Visualisation
* k-NN Accuracy Comparison for Different k Values

## How to Run

1. Download or clone this repository.
2. Install the required Python libraries:

```bash
pip install pandas matplotlib scikit-learn
```

3. Run the Decision Tree model:

```bash
python decision_tree_model.py
```

4. Run the k-NN model:

```bash
python knn_model.py
```

## Learning Outcomes

This project helped strengthen my understanding of:

* Machine learning classification
* Data cleaning and preprocessing
* Categorical encoding
* Feature scaling and normalisation
* Train-test dataset splitting
* Decision Tree and k-NN algorithms
* Model evaluation using multiple metrics
* Confusion matrix interpretation
* ROC curve and AUC evaluation
* Feature importance analysis
* Underfitting, overfitting, and model comparison

## Future Improvements

* Use one-hot encoding instead of manual label-style encoding for categorical variables
* Apply stratified train-test splitting to better handle class imbalance
* Compare more algorithms such as Logistic Regression, Random Forest, and SVM
* Add cross-validation for more reliable evaluation
* Apply hyperparameter tuning using GridSearchCV
* Save trained models using Joblib or Pickle
* Build a simple user interface for predictions
* Export evaluation results and charts automatically
