import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# load the dataset
data = pd.read_csv("/content/drive/MyDrive/DS 3000/nba_data_processed.csv")
data.head()

# drop all rows with empty values
data = data.dropna()

# using a threshold of points being 10+ or 20+ for realistic betting
# possibilites
data['PTS_10+'] = data['PTS'].apply(lambda x: 1 if x >= 10 else 0)
data['PTS_20+'] = data['PTS'].apply(lambda x: 1 if x >= 20 else 0)

X = data[['AST']]  # using assists as the single feature
y = data['PTS_10+']  # Target variable (change to the correct target column)

# function to train and evaluate a model for a given threshold
def train_and_evaluate_model(target, threshold):
    # split data using only 20% for training
    X_train, X_test, y_train, y_test = train_test_split(X, data[target], test_size=0.2, random_state=42)

    # initialize and train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # computing predictions
    y_pred = model.predict(X_test)

    # printing evaluation metrics
    print(f"Results for predicting if PTS >= {threshold}:\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("\n" + "="*50 + "\n")

# call function above to test out the thresholds
train_and_evaluate_model('PTS_10+', 10)
train_and_evaluate_model('PTS_20+', 20)

# **** Checking the correlation matrix to determine which feature pairs are good to pair
selected_columns = ["Age",'FG', 'FGA', '3PA', 'FT', 'MP', 'TRB', 'AST', 'STL', 'PTS']
correlation_matrix = data[selected_columns].corr()
# mask to hide diagonal elements (self-correlations)
mask = np.eye(len(correlation_matrix), dtype=bool)
masked_correlation_matrix = correlation_matrix.mask(mask)

# plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(masked_correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, square=True)
plt.title("Correlation Matrix (Selected Features)")
plt.show()

# **** CONFUSION MATRIX ****
def plot_confusion_matrix(y_test, y_pred, threshold):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for PTS >= {threshold}")
    plt.show()

# **** Feature importance Plot ****
def plot_feature_importance(model, features, threshold):
    importance = model.feature_importances_
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importance, y=features)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title(f"Feature Importance for PTS >= {threshold}")
    plt.show()

# **** ROC Curve ****
def plot_roc_curve(y_test, y_pred_prob, threshold):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for PTS >= {threshold}")
    plt.legend(loc="lower right")
    plt.show()

# an updated function to train and evaluate a model and generate visualizations based on the functions above
# confusion matrix, feature importance, and roc curve for each target
def train_and_evaluate_model_with_graphs(target, threshold):
    # same 20% training split
    X_train, X_test, y_train, y_test = train_test_split(X, data[target], test_size=0.2, random_state=42)

    # using random foresting for the classification problem
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # performing predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # displaying results
    print(f"Results for predicting if PTS >= {threshold}:\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("\n" + "=" * 50 + "\n")

    # VISUALIZATION
    # plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, threshold)
    # plot feature importance (if more than one feature is used)
    plot_feature_importance(model, X.columns, threshold)
    # plot ROC curve
    plot_roc_curve(y_test, y_pred_prob, threshold)

# perform evaluations with the 3 plots above
train_and_evaluate_model_with_graphs('PTS_10+', 10)
train_and_evaluate_model_with_graphs('PTS_20+', 20)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameter grid
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Output best parameters
print("Best Parameters:", grid_search.best_params_)

thresholds = [10, 15, 20, 25]
for threshold in thresholds:
    target = f'PTS_{threshold}+'
    data[target] = data['PTS'].apply(lambda x: 1 if x >= threshold else 0)
    train_and_evaluate_model_with_graphs(target, threshold)
