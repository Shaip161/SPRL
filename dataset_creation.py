import os
import json
import pandas as pd
import math
import nltk
from nltk.corpus import treebank
from nltk.corpus import propbank

def read_conllu(file_path):
    """
    This function reads a conllu file and returns a list of sentences.

    Parameters:
        file_path (str): Path to conllu file.

    Returns:
        list: List of sentences extracted from the conllu file.
    """
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        current_sentence = ""
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                if current_sentence:
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
                continue
            fields = line.split('\t')
            current_sentence = current_sentence + fields[1] + " "
        # Append the last sentence if it's not already appended
        if current_sentence:
            sentences.append(current_sentence.strip())
    return sentences

def create_white_json(white_data, train_sentences, test_sentences, dev_sentences):
    """
    This function creates JSON data for the white dataset based on the provided conllu data for train, test, and dev splits.

    Parameters:
        white_data (pandas DataFrame): Protoroles dataset.
        train_sentences (list): List of sentences (str) for the train split.
        test_sentences (list): List of sentences (str) for the test split.
        dev_sentences (list): List of sentences (str) for the dev split.

    Returns:
        dict: White data converted into JSON format.
    """
    json_data = {}
    for index, row in white_data.iterrows():
        # Note: an extremely rare javascript error produced null responses for 7 rows
        if math.isnan(row['Response']):
            print("Error")
            continue

        split = row['Split']
        pilot = row['Is.Pilot']
        sentence_loc, sentence_id_string = row['Sentence.ID'].split(" ")
        
        key = f"{split}-{pilot}-{sentence_id_string}"
        if key not in json_data:
            sentence_id = int(sentence_id_string) - 1
            if sentence_loc == "en-ud-train.conllu":
                sentence = train_sentences[sentence_id]
            elif sentence_loc == "en-ud-test.conllu":
                sentence = test_sentences[sentence_id]
            elif sentence_loc == "en-ud-dev.conllu":
                sentence = dev_sentences[sentence_id]

            words = sentence.split(" ")
            predicate = words[row['Pred.Token']]
            argument = ' '.join(words[row['Arg.Tokens.Begin']:row['Arg.Tokens.End'] + 1])

            json_data[key] = {
                'annotator': [],
                'applicable': [],
                'property': [],
                'ispilot': pilot,
                'label': [],
                'sentence': sentence,
                'split': row['Split'],
                'arg': argument,
                'pred': predicate,
                'gram_func': row['Gram.Func']
            }

        if row['Applicable'] == "yes":
            applic = True
        else:
            applic = False
             
        json_data[key]['annotator'].append(row['Annotator.ID'])
        json_data[key]['applicable'].append(applic)
        json_data[key]['property'].append(row['Property'])
        json_data[key]['label'].append(row['Response'])
    
    return json_data

def create_reisinger_json (reisinger_data):
    """
    Convert Reisinger data into a JSON format.

    This function takes Reisinger data in DataFrame format and converts it into a JSON format.
    The DataFrame should contain columns such as 'Split', 'Sentence.ID', 'Pred.Token', 'Arg.Pos',
    'Gram.Func', 'Roleset', 'Arg', 'Applicable', 'Property', and 'Response'. It generates a JSON
    object with each unique combination of 'Split' and 'Sentence.ID' as keys.

    Parameters:
    reisinger_data (pandas.DataFrame): A DataFrame containing Reisinger data.

    Returns:
    dict: A JSON-like dictionary containing the converted data.

    Raises:
    ValueError: If the DataFrame does not contain required columns or if 'Arg.Pos' format is invalid.
    FileNotFoundError: If the file paths for sentence parsing are incorrect.
    """
    json_data = {}
    for index, row in reisinger_data.iterrows():
        split = row['Split']

        key = f"{split}-{row['Sentence.ID']}"
        if key not in json_data:
            # Get sentence
            sentence_folder, sentence_id = row['Sentence.ID'].split('_')
            sentence_prefolder = sentence_folder[:2]

            current = os.getcwd()
            sentence_path = os.path.join(current, "wsj/" + sentence_prefolder + "/wsj_" + sentence_folder + ".mrg")
            tree = treebank.parsed_sents(sentence_path)[int(sentence_id)]
            sentence = ' '.join(tree.leaves())

            # Get predicate
            pred_token = row['Pred.Token']
            predicate = tree.leaves()[int(pred_token)]

            # Get argument(s)
            args = row['Arg.Pos'].split(',')
            argument = ""
            for arg in args:
                arg_token, arg_height = arg.split(':')
                tree_position = tree.leaf_treeposition(int(arg_token))

                for _ in range(int(arg_height) + 1):
                    tree_position = tree_position[:-1]

                argument_leaves = tree[tree_position].leaves()
                argument_phrase = ' '.join(argument_leaves)

                argument += argument_phrase + " "

            argument = argument.strip()

            # Create JSON object
            json_data[key] = {
                'applicable': [],
                'property': [],
                'label': [],
                'sentence': sentence,
                'split': split,
                'arg': argument,
                'pred': predicate,
                'gram_func': row['Gram.Func'],
                'roleset': row['Roleset'],
                'arg_num': row['Arg'],
            }
        
        json_data[key]['applicable'].append(row['Applicable'])
        json_data[key]['property'].append(row['Property'])
        json_data[key]['label'].append(row['Response'])
            
    return json_data

# Run
dirname = os.path.dirname(__file__)

train_path = os.path.join(dirname, "Universal Dependencies 1.2/ud-treebanks-v1.2/UD_English/en-ud-train.conllu")
test_path = os.path.join(dirname, "Universal Dependencies 1.2/ud-treebanks-v1.2/UD_English/en-ud-test.conllu")
dev_path = os.path.join(dirname, "Universal Dependencies 1.2/ud-treebanks-v1.2/UD_English/en-ud-dev.conllu")

train_sentences = read_conllu(train_path)
test_sentences = read_conllu(test_path)
dev_sentences = read_conllu(dev_path)

white_path = os.path.join(dirname, "datasets/protoroles_white/protoroles_eng_ud1.2_11082016.tsv")
white_data = pd.read_csv(white_path, delimiter="\t")

#reisinger_path = os.path.join(dirname, "datasets/protoroles_reisinger/protoroles_eng_pb/protoroles_eng_pb_08302015.tsv")
#reisinger_data = pd.read_csv(reisinger_path, delimiter="\t")

spr1_data = white_data[white_data['Protocol'] == 'spr1']
spr2_data = white_data[white_data['Protocol'] == 'spr2']
spr2_1_data = white_data[white_data['Protocol'] == 'spr2.1']

json_spr1 = create_white_json(spr1_data, train_sentences, test_sentences, dev_sentences)
json_spr2 = create_white_json(spr2_data, train_sentences, test_sentences, dev_sentences)
json_spr2_1 = create_white_json(spr2_1_data, train_sentences, test_sentences, dev_sentences)
json_spr2_no_pilot = {key: entry for key, entry in json_spr2.items() if entry.get('ispilot') == False}

#reisinger_json = create_reisinger_json(reisinger_data)

# Save JSON to a file
with open('datasets/json/spr1.json', 'w') as f:
    json.dump(json_spr1, f, indent=4)
with open('datasets/json/spr2.json', 'w') as f:
    json.dump(json_spr2, f, indent=4)
with open('datasets/json/spr2_1.json', 'w') as f:
    json.dump(json_spr2_1, f, indent=4)
with open('datasets/json/spr2_no_pilot.json', 'w') as f:
    json.dump(json_spr2_no_pilot, f, indent=4)
#with open('datasets/json/reis.json', 'w') as f:
#    json.dump(reisinger_json, f, indent=4)