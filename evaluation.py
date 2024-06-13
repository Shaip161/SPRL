import json
import numpy as np
import pickle
from sklearn.metrics import log_loss, f1_score, accuracy_score

Datasets_weights = {
    'SVM': 0,
    'log_reg': 0,
}

Datasets_bias = {
    'SVM': 0,
    'log_reg': 0,
}

def Read_Weights_Biases(file_path):
    """
    This function reads the weights and biases for SVM (Support Vector Machine) and Logistic Regression models from a specified file.

    Parameters:
    - file_path (str): The path to the file containing the weights and biases. 

    Returns:
    - svm_weights (list of float): The weights of the SVM model.
    - svm_bias (list of float): The bias of the SVM model. This should likely be a single float value rather than a list.
    - log_reg_weights (list of float): The weights of the Logistic Regression model.
    - log_reg_bias (list of float): The bias of the Logistic Regression model. This should likely be a single float value rather than a list.

    """

    # Initialize empty arrays to hold the weights
    svm_weights = []
    log_reg_weights = []
    svm_bias = 0
    log_reg_bias = 0

    # Read the file
    with open(file_path, 'r') as file:
       lines = file.readlines()

       # Assuming the second line contains SVM weights
       svm_weights_line = lines[1]
       svm_weights = [float(weight) for weight in svm_weights_line.split(": ")[1].split()]
       svm_bias_line = lines[2]
       svm_bias = [float(bias) for bias in svm_bias_line.split(": ")[1].split()]
    
       # Assuming the third line contains Logistic Regression weights
       log_reg_weights_line = lines[3]
       log_reg_weights = [float(weight) for weight in log_reg_weights_line.split(": ")[1].split()]
       log_reg_bias_line = lines[4]
       log_reg_bias = [float(bias) for bias in log_reg_bias_line.split(": ")[1].split()]
    
    return svm_weights, svm_bias, log_reg_weights, log_reg_bias


def WriteResults(filepath):
    """
    This function takes a filepath as input and writes the evaluation metrics of a classification model to a text file. 
    The metrics include the confusion matrix (both in counts and percentages), overall accuracy, F1 score, average log loss, 
    and class-specific accuracies for the 'Proto-Agent' and 'Proto-patient' classes. 

    Input:
    - filepath (str): The path of the file where the results will be written. If the file exists, it will be overwritten.

    """
    
    with open(filepath, 'w') as file:
        # Confusion Matrix + F1
        confusion_matrix, confusion_matrix_in_percent, F1, overall_accuracy = Calculate_Confusion_Matrix_F1_overall_accuracy(actual_predicted_labels)
        print('The Confusion Matrix : ', confusion_matrix, file=file)
        print('The Confusion Matrix in Percentage : ', confusion_matrix_in_percent, file=file)
        print('Overall accuracy : ', overall_accuracy, file=file)
        print('F1 : ', F1, file=file)

        # Calculate cumulative log loss
        average_log_loss = Calculate_cumulative_log_loss(actual_predicted_labels)
        print("Average Log Loss:", average_log_loss, file=file)

        # Calculate Class Specific Accuracy
        patient_accuracy, agent_accuracy = class_accuracy(actual_predicted_labels)
        print('Proto-Agent Class accuracy : ', agent_accuracy, file=file)
        print('Proto-patient Class accuracy : ', patient_accuracy, file=file)


# Sigmoid function
def sigmoid(x):
    """
    Calculates the sigmoid function.

    Parameters:
        x (float or np.array): The input value(s) for which the sigmoid function is to be calculated.

    Returns:
        float or np.array: The output of the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))

def getPredictedLabels(file_path, scaler_path, Training_type='SVM'):
    """
    Loads data from a JSON file and computes predicted labels based on pre-defined criteria and
    calculations involving dot products and the sigmoid function. The function filters data entries
    by prefixes and performs calculations to append actual labels and their corresponding sigmoid results
    to a list.

    Parameters:
        file_path (str): Path to the JSON file containing data entries.
        spr_type_start (str, optional): If specified, filters entries to those starting with this prefix.
            Defaults to None, in which case all predefined prefixes are considered.

    Returns:
        list: A list of lists, each containing an actual label and its corresponding sigmoid result.
    """
    actual_predicted_labels = []

    # Load the JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)

    test_data = []
    # Also take dev for test since we don't use it anywhere else
    for index, item in data.items():
        if item['split'] != 'train':
            test_data.append(item)

    X_test = []
    y_test = []

    for item in test_data:
        features = item['roberta_entailments'] + item['roberta_contradictions']
        X_test.append(features)
        y_test.append(item['is_agent'])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Load the scaler object from the file
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Scale entries
    X_test = scaler.transform(X_test)

    for x, y in zip(X_test, y_test):
        # Calculate dot products
        dot_product = np.dot(Datasets_weights[Training_type], x)

        # Read bias from Model
        bias = Datasets_bias[Training_type][0]

        # Sigmoid + Predicted Label
        sigmoid_result = sigmoid(dot_product + bias)
        actual_predicted_labels.append([y, sigmoid_result])
    """
    # Go through each Entry    
    for entry_key, entry_value in data.items():
        
        # Skip if not in the test split
        if entry_value["split"] == 'train':
            continue
        
        # Extract the required information
        is_agent = entry_value["is_agent"]
        features = entry_value['roberta_entailments'] + entry_value['roberta_contradictions']

        # Calculate dot products
        dot_product = np.dot(Datasets_weights[Training_type], features)
        
        # Read bias from Model
        bias = Datasets_bias[Training_type][0]

        # Sigmoid + Predicted Label
        sigmoid_result = sigmoid(dot_product + bias)
        actual_predicted_labels.append([is_agent, sigmoid_result])
    """

    return actual_predicted_labels
        
        
def Calculate_Confusion_Matrix_F1_overall_accuracy(actual_predicted_labels):
    """
    This function calculates the confusion matrix and the F1 score for a binary classification
    task from actual labels and predicted probabilities. 

    Parameters:
        actual_predicted_labels (list of tuples): A list where each element is a tuple containing
        the actual label (0 or 1) and the predicted probability (a float between 0 and 1) of the positive class.

    Returns:
        tuple: A tuple containing three elements:
            - The first element is the confusion matrix represented as a list of lists where
              the counts of true positive, false positive, false negative, and true negative predictions
              are recorded.
            - The second element is the confusion matrix in percentage terms.
            - The third element is the F1 score.
    """
    confusion_matrix = [[0, 0], [0, 0]]
    total_entries_evaluated  = 0

    for actual_label, sigmoid_result in actual_predicted_labels:
        predicted_label = 1 if sigmoid_result >= 0.5 else 0

        total_entries_evaluated += 1
            
        # Update Confusion Matrix
        if predicted_label == 1 and actual_label == 1:
            confusion_matrix[0][0] += 1
        elif predicted_label == 1 and actual_label == 0:
            confusion_matrix[0][1] += 1
        elif predicted_label == 0 and actual_label == 1:
            confusion_matrix[1][0] += 1
        elif predicted_label == 0 and actual_label == 0:
            confusion_matrix[1][1] += 1

    confusion_matrix_in_percent = [[0, 0],[0, 0]]

    for i in range(2):
        for j in range(2):
            confusion_matrix_in_percent[i][j] = confusion_matrix[i][j] / total_entries_evaluated

    if confusion_matrix[0][0] == 0:
        F1 = Precision = Recall = 0
    else:
        # Precision, Recall, F1
        Precision = confusion_matrix[0][0] / ( confusion_matrix[0][0] + confusion_matrix[0][1] )
        Recall = confusion_matrix[0][0] / ( confusion_matrix[0][0] + confusion_matrix[1][0] )
        F1 = 2 * ( Precision * Recall ) / ( Precision + Recall )
    
    accuracy = 0
    if confusion_matrix[0][0] == 0 and confusion_matrix[1][1]:
        accuracy = 0
    else:
        accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1])


    return confusion_matrix, confusion_matrix_in_percent, F1, accuracy


def Calculate_cumulative_log_loss(actual_predicted_labels):
    """
    This function calculates the cumulative log loss for a sequence of predictions in a binary classification task.
    It iterates through each prediction, calculating the log loss up to that point and averaging the cumulative
    log loss over all predictions. This provides insight into the model's performance over a sequence of predictions,
    allowing for the assessment of how well the model's probability estimates are calibrated.

    Parameters:
        actual_predicted_labels (list of tuples): A list where each element is a tuple containing
        the actual label (0 or 1) and the predicted probability (a float between 0 and 1).

    Returns:
        float: The average cumulative log loss over the sequence of predictions. 
    """
    actual_predicted_labels = np.array(actual_predicted_labels)
    # Extract actual labels and predicted probabilities
    actual_labels = actual_predicted_labels[:, 0].astype(int)  # Ensure actual labels are integers
    predicted_probs = actual_predicted_labels[:, 1]

    # Avoiding log(0) by clipping probabilities in a safe range; eps is a small number.
    eps = 1e-15
    predicted_probs = np.clip(predicted_probs, eps, 1 - eps)

    # Calculating log loss for each prediction
    log_losses = - (actual_labels * np.log(predicted_probs) +
                    (1 - actual_labels) * np.log(1 - predicted_probs))

    # Calculating the average log loss
    average_log_loss = np.mean(log_losses)

    return average_log_loss


def class_accuracy(actual_predicted_labels):
    """
    This function calculates the overall accuracy, as well as the class-specific accuracy
    for a binary classification task. It does this by comparing actual labels with predicted
    labels derived from sigmoid results.

    Parameters:
        actual_predicted_labels (list of tuples): A list where each element is a tuple containing
        the actual label (0 or 1) and the sigmoid result (a float between 0 and 1) of a prediction.

    Returns:
        tuple: A tuple containing three elements:
            - The first element is the accuracy for the class labeled as '0' (patient accuracy).
            - The second element is the accuracy for the class labeled as '1' (agent accuracy).
            - The third element is the overall accuracy, which is the proportion of correct predictions
            out of all predictions.
    """

    agent_total_accurate = 0
    agent_total = 0
    patient_total_accurate = 0
    patient_total = 0

    for actual_label, sigmoid_result in actual_predicted_labels:
        predicted_label = 1 if sigmoid_result >= 0.5 else 0
            
        if actual_label == 1:
            agent_total += 1 
            if predicted_label == 1:
                agent_total_accurate += 1
        
        if actual_label == 0:
            patient_total += 1
            if predicted_label == 0:
                patient_total_accurate += 1
    
    patient_accuracy = patient_total_accurate / patient_total if patient_total > 0 else 0
    agent_accuracy = agent_total_accurate / agent_total if agent_total > 0 else 0

    return patient_accuracy, agent_accuracy


# Read the Weight and Biases results from training file.
modelname = "roberta_reisinger"
file_path_weight_results = "evaluation_results/weights_and_bias/WB_" + modelname + ".txt" 
svm_weights, svm_bias, log_reg_weights, log_reg_bias = Read_Weights_Biases(file_path_weight_results)

# Update Dictionary
Datasets_weights['SVM'] = svm_weights
Datasets_bias['SVM'] = svm_bias
Datasets_weights['log_reg'] = log_reg_weights
Datasets_bias['log_reg'] = log_reg_bias

# Read Dataset to be evaluated + get prediction labels probablities
file_path_dataset = 'datasets/json/predictions/' + modelname +'_predictions.json'
scaler_path = 'evaluation_results/scaler/scaler_' + modelname + '.pkl'
actual_predicted_labels = getPredictedLabels(file_path_dataset, scaler_path, 'SVM')

# Write the results of the Evaluation to a file in results
file_path_eval_results = "evaluation_results/eval/svm/eval_svm_" + modelname + ".txt" 
WriteResults(file_path_eval_results)
