from utility import generate_hypothesis
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Hand crafted nonsensical senetences to Test our model for robustness

sentences = [
    # Proto-agent nonsensical sentences
    {"sentence": "The clock danced the morning away.", "predicate": "danced", "argument": "The clock", "role": "proto-agent", "comment": "Nonsensical action by inanimate object."},
    {"sentence": "The book whispered secrets to the wall.", "predicate": "whispered", "argument": "The book", "role": "proto-agent", "comment": "Inanimate objects in an impossible interaction."},
    {"sentence": "The sun played chess with the moon.", "predicate": "played", "argument": "The sun", "role": "proto-agent", "comment": "Celestial bodies engaging in human activities."},
    {"sentence": "The painting laughed at the artist.", "predicate": "laughed", "argument": "The painting", "role": "proto-agent", "comment": "Inanimate object displaying human emotions."},
    {"sentence": "The mountain challenged the sky to a race.", "predicate": "challenged", "argument": "The mountain", "role": "proto-agent", "comment": "Nature elements in unlikely scenarios."},
    {"sentence": "The software sang a lullaby.", "predicate": "sang", "argument": "The software", "role": "proto-agent", "comment": "Digital product performing human action."},
    {"sentence": "The coffee decided to be tea today.", "predicate": "decided", "argument": "The coffee", "role": "proto-agent", "comment": "Inanimate object making choices."},
    {"sentence": "The window dreamed of being a door.", "predicate": "dreamed", "argument": "The window", "role": "proto-agent", "comment": "Object with human-like desires."},
    {"sentence": "The shadow painted its own portrait.", "predicate": "painted", "argument": "The shadow", "role": "proto-agent", "comment": "Abstract concept performing an action."},
    {"sentence": "The road whispered tales to travelers.", "predicate": "whispered", "argument": "The road", "role": "proto-agent", "comment": "Inanimate entity engaging in communication."},
    
    # Proto-patient nonsensical sentences
    {"sentence": "The poem was read by the stars.", "predicate": "read", "argument": "The poem", "role": "proto-patient", "comment": "Celestial bodies engaging in human activity."},
    {"sentence": "The silence was broken by the light.", "predicate": "broken", "argument": "The silence", "role": "proto-patient", "comment": "Abstract concepts in an unlikely interaction."},
    {"sentence": "The idea was nurtured by the river.", "predicate": "nurtured", "argument": "The idea", "role": "proto-patient", "comment": "Natural elements interacting with abstract concepts."},
    {"sentence": "The laughter was carried by the wind.", "predicate": "carried", "argument": "The laughter", "role": "proto-patient", "comment": "Human emotion interacting with nature."},
    {"sentence": "The dance was taught by the shadows.", "predicate": "taught", "argument": "The dance", "role": "proto-patient", "comment": "Abstract concept as a teacher."},
    {"sentence": "The mystery was solved by the morning dew.", "predicate": "solved", "argument": "The mystery", "role": "proto-patient", "comment": "Natural phenomena engaging in detective work."},
    {"sentence": "The argument was settled by the paintings.", "predicate": "settled", "argument": "The argument", "role": "proto-patient", "comment": "Inanimate objects resolving human conflicts."},
    {"sentence": "The story was written by the clouds.", "predicate": "written", "argument": "The story", "role": "proto-patient", "comment": "Weather phenomena creating literature."},
    {"sentence": "The melody was composed by the forest.", "predicate": "composed", "argument": "The melody", "role": "proto-patient", "comment": "Ecosystem engaging in musical creation."},
    {"sentence": "The recipe was invented by the colors.", "predicate": "invented", "argument": "The recipe", "role": "proto-patient", "comment": "Abstract concepts contributing to culinary arts."}
]

#sätze in einer JSON file exporten.

REIS = ["changes_possession" ,"volition" ,"stationary" 
    ,"location_of_event" ,"existed_before" ,"awareness" 
    ,"exists_as_physical" ,"sentient" ,"existed_after" 
    ,"existed_during" ,"makes_physical_contact" ,"destroyed" 
    ,"change_of_location" ,"created" ,"instigation" 
    ,"manipulated_by_another" ,"change_of_state", "predicate_changed_argument"]

property_sets = {
    'reis_labelled': sorted(REIS),
}

cwd = os.getcwd()

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("models/roberta_white")
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large-mnli")

# Create the prediction set
prediction_set = {}
counter = 0
for s in sentences:
    sentence = s['sentence']
    arg = s['argument']
    pred = s['predicate']
    is_agent = 1 if s['role'] == 'proto-agent' else 0
    roberta_entailments = []
    roberta_contradictions = []
    
    # generate entry for prediction_set_bias
    for prop in property_sets["reis_labelled"]:
        hypothesis = generate_hypothesis(arg, pred, prop)
        inputs = tokenizer(sentence, hypothesis , return_tensors="pt")
        outputs = model(**inputs)
        roberta_entailments.append(outputs.logits[0][2].item())
        roberta_contradictions.append(outputs.logits[0][0].item())

    prediction_set[f'reis_labelled_{counter}'] = {
        'sentence' : sentence,
        'arg' : arg,
        'pred' : pred,
        'is_agent': is_agent,
        'roberta_entailments': roberta_entailments,
        'roberta_contradictions': roberta_contradictions,
        'split': 'test'
    }
    counter += 1


with open(f'datasets/json/predictions/OOD/OOD_roberta-white.json', 'w') as f:
    json.dump(prediction_set, f, indent=4)
