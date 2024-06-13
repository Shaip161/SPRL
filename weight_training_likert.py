import os
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from utility import average_labels_over_entries
cwd = os.getcwd()
dataset_json = "reis_labelled.json"
path = os.path.join(cwd, "datasets/json/" + dataset_json)

with open(path, 'r') as file:
    data = json.load(file)

averaged_data = average_labels_over_entries(data)

# Split train, test, dev
train_data = []
test_data = []
dev_data = []

for index, item in averaged_data.items():
    if item['split'] == 'train':
        train_data.append(item)
    elif item['split'] == 'test':
        test_data.append(item)
    elif item['split'] == 'dev':
        dev_data.append(item)
        
# Prepare data
# Train
X_train = []
y_train = []

for item in train_data:
    train_points = []
    for applic, label in zip(item['applicable'], item['label']):
        if applic:
            train_points.append(label)
        else:
            train_points.append(0)

    X_train.append(train_points)
    y_train.append(item['is_agent'])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Test
X_test = []
X_test_applicable = []
y_test = []

for item in test_data:
    test_points = []
    for applic, label in zip(item['applicable'], item['label']):
        if applic:
            test_points.append(label)
        else:
            test_points.append(0)

    X_test.append(test_points)
    y_test.append(item['is_agent'])

# I think we might as well use the dev stuff as well for testing
for item in dev_data:
    train_points = []
    for applic, label in zip(item['applicable'], item['label']):
        if applic:
            train_points.append(label)
        else:
            train_points.append(0)
            
    X_test.append(train_points)
    y_test.append(item['is_agent'])

X_test = np.array(X_test)
y_test = np.array(y_test)

# Scale features (Mean of 0 and standard deviation of 1 for every feature)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Support Vector Machine version
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train_scaled, y_train)

y_pred_svm = svm_clf.predict(X_test_scaled) # Evaluate

accuracy = accuracy_score(y_test, y_pred_svm)
print('SVM test accuracy:', accuracy)

weights = svm_clf.coef_
print('SVM weights:', weights)

# Log Reg version
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)

y_pred_log_reg = log_reg.predict(X_test_scaled) # Evaluate

accuracy = accuracy_score(y_test, y_pred_log_reg)
print('Log Reg test accuracy:', accuracy)

weights = log_reg.coef_
print('Log Reg weights:', weights)
