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

data = pd.read_csv("/content/drive/MyDrive/DS 3000/nba_data_processed.csv")

# Calculate correlations
data.head()

# Drop rows with missing values (if any)
data = data.dropna()

# Define thresholds and create binary labels for each threshold
data['PTS_10+'] = data['PTS'].apply(lambda x: 1 if x >= 10 else 0)
data['PTS_20+'] = data['PTS'].apply(lambda x: 1 if x >= 20 else 0)

# Define the feature and target for each threshold
X = data[['AST']]  # Use assists as the single feature

# Function to train and evaluate a model for a given threshold
def train_and_evaluate_model(target, threshold):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, data[target], test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print(f"Results for predicting if PTS >= {threshold}:\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("\n" + "="*50 + "\n")

# Train and evaluate models for each threshold
train_and_evaluate_model('PTS_10+', 10)
train_and_evaluate_model('PTS_20+', 20)

selected_columns = ["Age",'FG', 'FGA', '3PA', 'FT', 'MP', 'TRB', 'AST', 'STL', 'PTS']
correlation_matrix = data[selected_columns].corr()

# Mask to hide diagonal elements (self-correlations)
mask = np.eye(len(correlation_matrix), dtype=bool)
masked_correlation_matrix = correlation_matrix.mask(mask)

# Plot the masked correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(masked_correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, square=True)
plt.title("Correlation Matrix (Selected Features)")
plt.show()

# Function to plot the confusion matrix
def plot_confusion_matrix(y_test, y_pred, threshold):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for PTS >= {threshold}")
    plt.show()

# Function to plot feature importance
def plot_feature_importance(model, features, threshold):
    importance = model.feature_importances_
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importance, y=features)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title(f"Feature Importance for PTS >= {threshold}")
    plt.show()

# Function to plot the ROC Curve
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

# Updated function to train and evaluate a model and generate visualizations
def train_and_evaluate_model_with_graphs(target, threshold):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, data[target], test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    print(f"Results for predicting if PTS >= {threshold}:\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("\n" + "=" * 50 + "\n")

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, threshold)

    # Plot feature importance (if more than one feature is used)
    plot_feature_importance(model, X.columns, threshold)

    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_prob, threshold)

# Train and evaluate models for each threshold with visualizations
train_and_evaluate_model_with_graphs('PTS_10+', 10)
train_and_evaluate_model_with_graphs('PTS_20+', 20)