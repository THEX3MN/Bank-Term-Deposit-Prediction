# Bank Term Deposit Prediction using Machine Learning

A Python machine learning project developed as part of an academic assignment to predict whether a bank customer will subscribe to a term deposit based on customer information, campaign details, and previous marketing interactions.

The project applies supervised machine learning techniques using two classification models: **Decision Tree** and **k-Nearest Neighbours (k-NN)**. It includes data cleaning, categorical encoding, feature scaling, train-test splitting, model training, performance evaluation, and visualisation of results.

## Project Overview

Banks often use direct marketing campaigns, such as phone calls, to promote term deposit subscriptions. This project uses the Bank Marketing dataset to predict whether a customer is likely to subscribe to a term deposit.

The target variable is:

* `yes` – the customer subscribed to a term deposit
* `no` – the customer did not subscribe to a term deposit

The main goal of this project is to compare machine learning models and evaluate their performance using appropriate classification metrics.

## Dataset

The project uses the Bank Marketing dataset, which contains customer and campaign-related attributes such as:

* Age
* Job
* Marital status
* Education level
* Default status
* Housing loan
* Personal loan
* Contact type
* Month and day of contact
* Call duration
* Campaign contact count
* Previous campaign outcome
* Economic indicators
* Target variable: `y`

## Machine Learning Models

### Decision Tree Classifier

The Decision Tree model was used to classify whether a customer would subscribe to a term deposit. This model was selected because it is easy to interpret and can show how different features contribute to classification decisions.

The model includes:

* Data cleaning and preprocessing
* Manual categorical encoding
* Train-test splitting
* Model training and prediction
* Accuracy, precision, recall, and F1 score evaluation
* Confusion matrix
* ROC curve and AUC score
* Feature importance visualisation
* Decision tree structure visualisation

### k-Nearest Neighbours Classifier

The k-NN model was used as a distance-based classification algorithm. Since k-NN is sensitive to feature scale, Min-Max normalisation was applied before training the model.

The model includes:

* Data cleaning and preprocessing
* Manual categorical encoding
* Feature normalisation using Min-Max scaling
* Train-test splitting
* Model training and prediction
* Accuracy, precision, recall, and F1 score evaluation
* Confusion matrix
* Feature space visualisation
* Accuracy comparison using different k values

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

1. Loaded and inspected the Bank Marketing dataset
2. Checked for missing values and duplicate records
3. Removed duplicate rows and missing values
4. Encoded categorical variables into numerical values
5. Separated input features and target variable
6. Applied feature scaling for the k-NN model
7. Split the dataset into training and testing sets
8. Trained Decision Tree and k-NN models
9. Generated predictions using the test dataset
10. Evaluated models using classification metrics
11. Created visualisations to support model interpretation

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC Curve
* AUC Score

Since the dataset is imbalanced, accuracy alone may not fully represent model performance. Precision, recall, F1 score, confusion matrix, and AUC score were also considered to better evaluate the models.

## Screenshots and Results

### Data Loading Preview

![Data Loading Preview](Data%20loading%20preview.png)

### Decision Tree Metrics

![Decision Tree Metrics](Decision%20tree%20metrics.png)

### Decision Tree ROC Curve

![Decision Tree ROC Curve](Decision%20tree%20ROC%20curve.png)

### Decision Tree Feature Importance

![Decision Tree Feature Importance](Decision%20tree%20feature%20importance.png)

### k-NN Metrics

![k-NN Metrics](k-NN%20metrics.png)

### k-NN Feature Space Visualisation

![k-NN Feature Space](k-NN%20feature%20space.png)

### k-NN Accuracy vs k

![k-NN Accuracy vs k](k-NN%20accuracy%20vs%20k.png)

## How to Run

1. Download or clone this repository.
2. Open the project folder.
3. Install the required Python libraries:

```bash
pip install pandas matplotlib scikit-learn
```

4. Run the Decision Tree model:

```bash
python decision_tree_model.py
```

5. Run the k-NN model:

```bash
python knn_model.py
```

## Learning Outcomes

This project helped strengthen my understanding of:

* Supervised machine learning classification
* Data cleaning and preprocessing
* Categorical encoding
* Feature scaling and normalisation
* Train-test splitting
* Decision Tree and k-NN algorithms
* Model evaluation using multiple metrics
* Confusion matrix interpretation
* ROC curve and AUC analysis
* Feature importance interpretation
* Underfitting, overfitting, and model comparison

## Future Improvements

* Use one-hot encoding for categorical variables
* Apply stratified train-test splitting to better handle class imbalance
* Compare additional models such as Logistic Regression, Random Forest, and SVM
* Use cross-validation for more reliable evaluation
* Apply hyperparameter tuning with GridSearchCV
* Save trained models using Joblib or Pickle
* Build a simple user interface for real-time prediction
* Export model results and charts automatically
