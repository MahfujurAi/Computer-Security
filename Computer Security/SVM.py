import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score,roc_auc_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_nbaiot(filename):
    return np.loadtxt(
        os.path.join("/kaggle/input/nbaiot-dataset", filename),
        delimiter=",",
        skiprows=1
    )

# Load the benign traffic data
benign = load_nbaiot("1.benign.csv")

# Split the benign traffic data into training and testing sets
X_train_benign, X_test_benign = train_test_split(benign, test_size=0.2, random_state=42)

# Load the malicious traffic data
mirai_scan = load_nbaiot("1.mirai.scan.csv")
mirai_ack = load_nbaiot("1.mirai.ack.csv")
mirai_syn = load_nbaiot("1.mirai.syn.csv")
mirai_udp = load_nbaiot("1.mirai.udp.csv")
mirai_udpplain = load_nbaiot("1.mirai.udpplain.csv")

# Combine the malicious traffic data into a single dataset
malicious = np.concatenate([mirai_scan, mirai_ack, mirai_syn, mirai_udp, mirai_udpplain])

# Create labels for the benign and malicious traffic data
y_train_benign = np.zeros(len(X_train_benign))
y_test_benign = np.zeros(len(X_test_benign))
y_train_malicious = np.ones(len(malicious))

# Split the benign and malicious traffic data into training and testing sets
X_train_malicious, X_test_malicious, y_train_malicious, y_test_malicious = train_test_split(malicious, y_train_malicious, test_size=0.2, random_state=42)

# Combine the benign and malicious data into training and testing sets
X_train = np.concatenate([X_train_benign, X_train_malicious])
X_test = np.concatenate([X_test_benign, X_test_malicious])
y_train = np.concatenate([y_train_benign, y_train_malicious])
y_test = np.concatenate([y_test_benign, y_test_malicious])

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "solver": ["lbfgs", "liblinear"],
    "penalty": ["l1", "l2"],
    "max_iter": [100, 200, 500],
    "kernel": ["linear", "rbf"],
    "gamma": [0.001, 0.01, 0.1],
}

model1 = SVC(random_state=42)
model = GridSearchCV(model1, param_grid, cv=5, scoring=["accuracy", "f1_macro", "roc_auc"], refit="accuracy")

model.fit(X_train, y_train)

# Print the best parameter and the best score
print(f"Best parameter: {model.best_params_}")
print(f"Best score: {model.best_score_}")

# Evaluate the model on the validation set
val_preds = model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation accuracy: {val_acc}")

# Calculate and print the validation loss
val_loss = log_loss(y_val, val_preds)
print(f"Validation loss: {val_loss}")


# Evaluate the model on the validation set
val_preds = model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
val_f1 = f1_score(y_val, val_preds)
val_auc = roc_auc_score(y_val, val_preds)
print(f"Validation accuracy: {val_acc}")
print(f"Validation F1 score: {val_f1}")
print(f"Validation AUC-ROC: {val_auc}")

# Evaluate the model on the test set
test_preds = model.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds)
test_auc = roc_auc_score(y_test, test_preds)
print(f"Test accuracy: {test_acc}")
print(f"Test F1 score: {test_f1}")
print(f"Test AUC-ROC: {test_auc}")

# Plot the validation and test accuracy graph
plt.plot(val_acc) # plot the validation accuracy
plt.plot(test_acc) # plot the test accuracy
plt.title('Validation and Test Accuracy')
plt.xlabel('Validation and Test Set')
plt.ylabel('Accuracy')
plt.legend(['Validation', 'Test'])
plt.show() # show the graph

# Plot the validation and test F1 score graph
plt.plot(val_f1) # plot the validation F1 score
plt.plot(test_f1) # plot the test F1 score
plt.title('Validation and Test F1 Score')
plt.xlabel('Validation and Test Set')
plt.ylabel('F1 Score')
plt.legend(['Validation', 'Test'])
plt.show() # show the graph

# Plot the validation and test AUC-ROC graph
plt.plot(val_auc) # plot the validation AUC-ROC
plt.plot(test_auc) # plot the test AUC-ROC
plt.title('Validation and Test AUC-ROC')
plt.xlabel('Validation and Test Set')
plt.ylabel('AUC-ROC')
plt.legend(['Validation', 'Test'])
plt.show() # show the graph

# Create a heatmap of the confusion matrix
sns.heatmap(val_cm, annot=True, cmap="Blues", fmt="g")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Validation Confusion Matrix")

# Save the heatmap as a PNG file
plt.savefig("val.png", dpi=300, format="png")


# Create a heatmap of the confusion matrix
sns.heatmap(test_cm, annot=True, cmap="Greens", fmt="g")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Testing Confusion Matrix")

# Save the heatmap as a PNG file
plt.savefig("testing_confusion_matrix_heatmap.png", dpi=300, format="png")