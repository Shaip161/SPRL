import os
import json
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from utility import average_labels_over_entries
cwd = os.getcwd()
modelname = "roberta_averaged"

path = os.path.join(cwd, "datasets/json/predictions/" + modelname + "_predictions.json")

with open(path, 'r') as file:
    data = json.load(file)

# Split train, test, dev
train_data = []
test_data = []
dev_data = []

for index, item in data.items():
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
    features = item['roberta_entailments'] + item['roberta_contradictions']
    X_train.append(features)
    y_train.append(item['is_agent'])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Test
X_test = []
y_test = []

for item in test_data:
    features = item['roberta_entailments'] + item['roberta_contradictions']
    X_test.append(features)
    y_test.append(item['is_agent'])

# I think we might as well use the dev stuff as well for testing
for item in dev_data:
    features = item['roberta_entailments'] + item['roberta_contradictions']
    X_test.append(features)
    y_test.append(item['is_agent'])

X_test = np.array(X_test)
y_test = np.array(y_test)

# Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Export Scaler
with open('evaluation_results/scaler/scaler_' + modelname + '.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Support Vector Machine version
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train, y_train)

y_pred_svm = svm_clf.predict(X_test) # Evaluate

accuracy = accuracy_score(y_test, y_pred_svm)
print("Accuracy SVM:", accuracy)

svm_weights = svm_clf.coef_[0]
svm_bias = svm_clf.intercept_

# Log Reg version
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred_log_reg = log_reg.predict(X_test) # Evaluate

accuracy = accuracy_score(y_test, y_pred_log_reg)
print("Accuracy log reg:", accuracy)

log_reg_weights = log_reg.coef_[0]
log_reg_bias = log_reg.intercept_

# Add necessary results to a txt file
content = f"Feature File: {modelname}\n"
content += "SVM Weights: " + " ".join(map(str, svm_weights)) + "\n"
content += "SVM Bias: " + " ".join(map(str, svm_bias)) + "\n"
content += "Log Reg Weights: " + " ".join(map(str, log_reg_weights)) + "\n"
content += "Log Reg Bias: " + " ".join(map(str, log_reg_bias)) + "\n"

new_file_path = os.path.join(cwd, "evaluation_results/weights_and_bias/WB_" + modelname + ".txt")

with open(new_file_path, 'w') as file:
    file.write(content)
