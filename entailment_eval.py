import os
from utility import generate_hypothesis, average_labels_over_entries, read_json_data
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

cwd = os.getcwd()

# Load the model
model_name = "roberta_non-averaged"
model = AutoModelForSequenceClassification.from_pretrained("models/" + model_name)
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large-mnli")

dirname = os.path.dirname(__file__)
reis_path = os.path.join(dirname, "datasets/json/reis.json")

with open(reis_path, 'r') as json_file:
    reis_data = json.load(json_file)

averaged_reis = average_labels_over_entries(reis_data)

train_df, test_df, dev_df = read_json_data(averaged_reis)

accuracy = 0
accuracy_cont = 0
accuracy_cont_total = 0
accuracy_neut = 0
accuracy_neut_total = 0
accuracy_enta = 0
accuracy_enta_total = 0
total = 0

counter = 0
for key, item in test_df.iterrows():
    counter +=1
    if counter % 100 == 1:
        print(counter)
    
    premise = item['premise']
    hypothesis = item['hypothesis']
    label = item['label']
    
    inputs = tokenizer(premise, hypothesis , return_tensors="pt")
    outputs = model(**inputs)

    max_var = outputs.logits[0][2].item()
    max_ind = 2
    if max_var < outputs.logits[0][1].item():
        max_var = outputs.logits[0][1].item()
        max_ind = 1
    if max_var < outputs.logits[0][0].item():
        max_var = outputs.logits[0][0].item()
        max_ind = 0

    if max_ind == label:
        accuracy += 1
        if label == 0:
            accuracy_cont += 1
            accuracy_cont_total += 1
        elif label == 1:
            accuracy_neut += 1
            accuracy_neut_total += 1
        elif label == 2:
            accuracy_enta += 1
            accuracy_enta_total += 1
    else:
        if label == 0:
            accuracy_cont_total += 1
        elif label == 1:
            accuracy_neut_total += 1
        elif label == 2:
            accuracy_enta_total += 1
    
    total += 1

accuracy_overall = accuracy / total
accuracy_cont_overall = accuracy_cont / accuracy_cont_total
accuracy_neut_overall = accuracy_neut / accuracy_neut_total
accuracy_enta_overall = accuracy_enta / accuracy_enta_total

with open('overall_accuracy_non-averaged', 'w') as file:
    # Confusion Matrix + F1
    print('overall accuracy : ', accuracy_overall, file=file)
    print('overall accuracy cont : ', accuracy_cont_overall, file=file)
    print('overall accuracy neut : ', accuracy_neut_overall, file=file)
    print('overall accuracy enta: ', accuracy_enta_overall, file=file)
    print('total : ', total, file=file)
    print('cont total : ', accuracy_cont_total, file=file)
    print('neut total : ', accuracy_neut_total, file=file)
    print('enta total : ', accuracy_enta_total, file=file)