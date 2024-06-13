import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utility import generate_hypothesis

REIS = ["changes_possession" ,"volition" ,"stationary" 
    ,"location_of_event" ,"existed_before" ,"awareness" 
    ,"exists_as_physical" ,"sentient" ,"existed_after" 
    ,"existed_during" ,"makes_physical_contact" ,"destroyed" 
    ,"change_of_location" ,"created" ,"instigation" 
    ,"manipulated_by_another" ,"change_of_state", "predicate_changed_argument"]

S2 = ["awareness","change_of_location","change_of_possession",
    "change_of_state","change_of_state_continuous","existed_after",
    "existed_before","existed_during","instigation",
    "partitive","sentient","volition",
    "was_for_benefit","was_used"]

S1 = ["awareness",
            "change_of_location",
            "change_of_state",
            "changes_possession",
            "existed_after",
            "existed_before",
            "existed_during",
            "exists_as_physical",
            "instigation",
            "location_of_event",
            "makes_physical_contact",
            "was_used",
            "predicate_changed_argument",
            "sentient",
            "stationary",
            "volition"]

property_sets = {
    'reis_labelled': sorted(REIS),
    'spr1': sorted(S1),
    'spr2': sorted(S2),
}

cwd = os.getcwd()

# Load the model
modelname = "roberta_averaged"
model = AutoModelForSequenceClassification.from_pretrained("models/" + modelname)
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large-mnli")

# Load the data
spr_type = "reis_labelled"
data_path = os.path.join(cwd, "datasets/json/" + spr_type + ".json")
with open(data_path, "r") as f:
    data = json.load(f)

# Create the prediction set
prediction_set = {}
counter = 0
for key, item in data.items():
    if counter % 100 == 0:
        print(str(counter) + " of 2719.")
    counter += 1
    
    sentence = item['sentence']
    arg = item['arg']
    pred = item['pred']
    is_agent = item['is_agent']
    roberta_entailments = []
    roberta_contradictions = []
    
    # generate entry for prediction_set
    for prop in property_sets[spr_type]:
        hypothesis = generate_hypothesis(arg, pred, prop)
        inputs = tokenizer(sentence, hypothesis , return_tensors="pt")
        outputs = model(**inputs)
        roberta_entailments.append(outputs.logits[0][2].item())
        roberta_contradictions.append(outputs.logits[0][0].item())

    prediction_set[key] = {
        'sentence' : sentence,
        'arg' : arg,
        'pred' : pred,
        'is_agent': is_agent,
        'roberta_entailments': roberta_entailments,
        'roberta_contradictions': roberta_contradictions,
        'split': item['split']
    }


        
with open(f'datasets/json/predictions/{modelname}_predictions.json', 'w') as f:
    json.dump(prediction_set, f, indent=4)
