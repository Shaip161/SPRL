import json
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from utility import generate_hypothesis, average_labels_over_entries, read_json_data

class CustomDataset(Dataset):
    def __init__(self, data, maxlen, with_labels=True, roberta_model='FacebookAI/roberta-large-mnli'):

        self.data = data  # pandas dataframe
        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(roberta_model)  

        self.maxlen = maxlen
        self.with_labels = with_labels 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting premise and hypothesis at the specified index in the data frame
        sent1 = str(self.data.loc[index, 'premise'])
        sent2 = str(self.data.loc[index, 'hypothesis'])

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent1, sent2, 
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt',
                                      return_token_type_ids=True)  # Return torch.Tensor objects
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        encodings = {'input_ids': token_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attn_masks
                }

        if self.with_labels:  # True if the dataset has labels
            encodings['labels'] = torch.tensor(self.data.loc[index, 'label'])
            return encodings  
        else:
            return token_ids, attn_masks, token_type_ids
        


# Run
dirname = os.path.dirname(__file__)
reis_path = os.path.join(dirname, "datasets/json/reis.json")
spr1_path = os.path.join(dirname, "datasets/json/spr1.json")
spr2_path = os.path.join(dirname, "datasets/json/spr2.json")
spr2_1_path = os.path.join(dirname, "datasets/json/spr2_1.json")
spr2_no_pilot_path = os.path.join(dirname, "datasets/json/spr2_no_pilot.json")

with open(reis_path, 'r') as json_file:
    reis_data = json.load(json_file)
with open(spr1_path, 'r') as json_file:
    spr1_data = json.load(json_file)
with open(spr2_path, 'r') as json_file:
    spr2_data = json.load(json_file)
with open(spr2_1_path, 'r') as json_file:
    spr2_1_data = json.load(json_file)
with open(spr2_no_pilot_path, 'r') as json_file:
    spr2_no_pilot_data = json.load(json_file)

averaged_reis = average_labels_over_entries(reis_data)
averaged_spr1 = average_labels_over_entries(spr1_data)
averaged_spr2 = average_labels_over_entries(spr2_data)
averaged_spr2_1 = average_labels_over_entries(spr2_1_data)
averaged_spr2_no_pilot = average_labels_over_entries(spr2_no_pilot_data)

# Combine the data
combined_data = {**averaged_spr1, **averaged_spr2, **averaged_spr2_1, **averaged_reis}

train_df, test_df, dev_df = read_json_data(averaged_reis)

# Setup
maxlen = 128
train_dataset = CustomDataset(train_df, maxlen)
dev_dataset = CustomDataset(dev_df, maxlen)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

model = AutoModelForSequenceClassification.from_pretrained('FacebookAI/roberta-large-mnli')

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=dev_dataset             # evaluation dataset
)

trainer.train()

output_path = os.path.join(dirname, "models/roberta_reisinger")
trainer.save_model(output_path)