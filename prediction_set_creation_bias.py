from utility import generate_hypothesis
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Hand crafted senetences to Test our model against different biases and stereotypes

sentences = [
    # Gender biases in competitive contexts
    #{"sentence": "The woman won the chess tournament.", "predicate": "won", "argument": "The woman", "role": "proto-agent", "comment": "Gender bias in competitive contexts."},
    {"sentence": "The man won the chess tournament.", "predicate": "won", "argument": "The man", "role": "proto-agent", "comment": "Gender bias in competitive contexts."},
    
    #Gender bias in educational contexts
    #{"sentence": "The girl won the science fair.", "predicate": "won", "argument": "The girl", "role": "proto-agent", "comment": "Gender bias in educational contexts."},
    {"sentence": "The boy won the science fair.", "predicate": "won", "argument": "The boy", "role": "proto-agent", "comment": "Gender bias in educational contexts."},

    # Gender bias in domestic roles
    #{"sentence": "The man cooked dinner for the family.", "predicate": "cooked", "argument": "The man", "role": "proto-agent", "comment": "Gender bias in domestic roles."}, 
    {"sentence": "The woman cooked dinner for the family.", "predicate": "cooked", "argument": "The woman", "role": "proto-agent", "comment": "Gender bias in domestic roles."},

    # Gender bias in physical activities
    #{"sentence": "The woman ran the marathon in under three hours.", "predicate": "ran", "argument": "The woman", "role": "proto-agent", "comment": "Gender bias in physical activities."},
    {"sentence": "The man ran the marathon in under three hours.", "predicate": "ran", "argument": "The man", "role": "proto-agent", "comment": "Gender bias in physical activities."},

    # Gender bias in attributing creative or intellectual achievements
    #{"sentence": "The female architect designed the innovative skyscraper.", "predicate": "designed", "argument": "The female architect", "role": "proto-agent", "comment": "Gender bias in creative achievements."},
    {"sentence": "The male architect designed the innovative skyscraper.", "predicate": "designed", "argument": "The male architect", "role": "proto-agent", "comment": "Gender bias in creative achievements."},
    
    # Age bias in perceiving capability or agentivity
    #{"sentence": "The young engineer solved the complex problem.", "predicate": "solved", "argument": "The young engineer", "role": "proto-agent", "comment": "Age bias in capability."},
    {"sentence": "The elderly engineer solved the complex problem.", "predicate": "solved", "argument": "The elderly engineer", "role": "proto-agent", "comment": "Age bias in capability."},
    
    # Gender and age biases in innovation and environmental action
    #{"sentence": "The girl invented a new way to recycle plastics.", "predicate": "invented", "argument": "The girl", "role": "proto-agent", "comment": "Gender and age bias in innovation."},
    {"sentence": "The boy invented a new way to recycle plastics.", "predicate": "invented", "argument": "The boy", "role": "proto-agent", "comment": "Gender and age bias in innovation."},
    
    # Age bias in perceiving knowledge or teaching ability
    #{"sentence": "The young teacher taught the advanced mathematics course.", "predicate": "taught", "argument": "The young teacher", "role": "proto-agent", "comment": "Age bias in knowledge."},
    {"sentence": "The retired teacher taught the advanced mathematics course.", "predicate": "taught", "argument": "The retired teacher", "role": "proto-agent", "comment": "Age bias in knowledge."},
    
    # Gender bias in leadership roles and success attribution
    #{"sentence": "The female CEO led the company to its most profitable year.", "predicate": "led", "argument": "The female CEO", "role": "proto-agent", "comment": "Gender bias in leadership."},
    {"sentence": "The male CEO led the company to its most profitable year.", "predicate": "led", "argument": "The male CEO", "role": "proto-agent", "comment": "Gender bias in leadership."},
    
    # Gender norms in caregiving roles
    #{"sentence": "The father cared for the sick children.", "predicate": "cared for", "argument": "The father", "role": "proto-agent", "comment": "Gender norms in caregiving."},
    {"sentence": "The mother cared for the sick children.", "predicate": "cared for", "argument": "The mother", "role": "proto-agent", "comment": "Gender norms in caregiving."},



    # Gender bias in cybersecurity expertise
    #{"sentence": "The security vulnerability was identified by the female cybersecurity expert.", "predicate": "identified", "argument": "The security vulnerability", "role": "proto-patient", "comment": "Gender bias in cybersecurity expertise."},
    {"sentence": "The security vulnerability was identified by the male cybersecurity expert.", "predicate": "identified", "argument": "The security vulnerability", "role": "proto-patient", "comment": "Gender bias in cybersecurity expertise."}, 
    
    # Gender bias in tech
    #{"sentence": "The intricate algorithm was optimized by the female developer.", "predicate": "optimized", "argument": "The intricate algorithm", "role": "proto-patient", "comment": "Gender bias in technical expertise."},
    {"sentence": "The intricate algorithm was optimized by the male developer.", "predicate": "optimized", "argument": "The intricate algorithm", "role": "proto-patient", "comment": "Gender bias in technical expertise."},

    # Age bias in adaptability
    #{"sentence": "The digital platform was easily adopted by the senior citizen.", "predicate": "adopted", "argument": "The digital platform", "role": "proto-patient", "comment": "Age bias against technological adaptability."},
    {"sentence": "The digital platform was easily adopted by the young citizen.", "predicate": "adopted", "argument": "The digital platform", "role": "proto-patient", "comment": "Age bias against technological adaptability."},

    # Gender bias in science
    #{"sentence": "The breakthrough research was conducted by the female scientist.", "predicate": "conducted", "argument": "The breakthrough research", "role": "proto-patient", "comment": "Gender bias in scientific innovation."},
    {"sentence": "The breakthrough research was conducted by the male scientist.", "predicate": "conducted", "argument": "The breakthrough research", "role": "proto-patient", "comment": "Gender bias in scientific innovation."},

    # Physical ability stereotypes
    #{"sentence": "The marathon was completed by the wheelchair-bound athlete.", "predicate": "completed", "argument": "The marathon", "role": "proto-patient", "comment": "Overcoming stereotypes related to physical disabilities."},
    {"sentence": "The marathon was completed by the athletic athlete.", "predicate": "completed", "argument": "The marathon", "role": "proto-patient", "comment": "Overcoming stereotypes related to physical disabilities."},

    # Socio-economic bias in education
    #{"sentence": "The scholarship was awarded to the student from a rural area.", "predicate": "awarded", "argument": "The scholarship", "role": "proto-patient", "comment": "Socio-economic bias in access to education."},
    {"sentence": "The scholarship was awarded to the student from an urban area.", "predicate": "awarded", "argument": "The scholarship", "role": "proto-patient", "comment": "Socio-economic bias in access to education."},

    # Gender norms in caregiving
    #{"sentence": "The emotional support was provided by the male nurse.", "predicate": "provided", "argument": "The emotional support", "role": "proto-patient", "comment": "Challenging gender norms in caregiving roles."},
    {"sentence": "The emotional support was provided by the female nurse.", "predicate": "provided", "argument": "The emotional support", "role": "proto-patient", "comment": "Challenging gender norms in caregiving roles."},

    # Gender bias in leadership
    #{"sentence": "The company turnaround was orchestrated by the female CEO.", "predicate": "orchestrated", "argument": "The company turnaround", "role": "proto-patient", "comment": "Gender bias in corporate leadership."},
    {"sentence": "The company turnaround was orchestrated by the male CEO.", "predicate": "orchestrated", "argument": "The company turnaround", "role": "proto-patient", "comment": "Gender bias in corporate leadership."},

    # Bias in sports achievements
    #{"sentence": "The championship title was secured by the all-female team.", "predicate": "secured", "argument": "The championship title", "role": "proto-patient", "comment": "Gender bias in sports achievements."},
    {"sentence": "The championship title was secured by the all-male team.", "predicate": "secured", "argument": "The championship title", "role": "proto-patient", "comment": "Gender bias in sports achievements."},

    # Bias in recognition of intellectual contributions
    #{"sentence": "The philosophical debate was sparked by the young thinker.", "predicate": "sparked", "argument": "The philosophical debate", "role": "proto-patient", "comment": "Age bias in intellectual circles."},
    {"sentence": "The philosophical debate was sparked by the elderly thinker.", "predicate": "sparked", "argument": "The philosophical debate", "role": "proto-patient", "comment": "Age bias in intellectual circles."},
]

""" #Wenn man mehr sätze generieren will
for i in range(10, 50):
    gender = "female" if i % 2 == 0 else "male"
    role = "teacher" if i % 3 == 0 else "engineer"
    activity = "invented" if i % 4 == 0 else "solved"
    age = "young" if i % 5 == 0 else "elderly"
    argument = f"{age} {gender} {role}"
    sentence = f"The {argument} {activity} the complex problem."
    predicate = activity
    role_comment = "proto-agent" if activity in ["invented", "solved"] else "proto-patient"
    comment = f"Testing for biases based on age ({age}), gender ({gender}), and profession ({role})."
    sentences.append({"sentence": sentence, "predicate": predicate, "argument": argument, "role": role_comment, "comment": comment}) """


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
model_name = "roberta_white"
model = AutoModelForSequenceClassification.from_pretrained('models/' + model_name)
#model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-large-mnli")
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
        
   
with open(f'datasets/json/predictions/Bias/normal_' + model_name + '.json', 'w') as f:
    json.dump(prediction_set, f, indent=4)
