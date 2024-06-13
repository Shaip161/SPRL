from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import MultiLabelClassificationExplainer

# Load model and tokenizer
model_name = "models/roberta_averaged"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large-mnli")

# Initialize explainer
cls_explainer = MultiLabelClassificationExplainer(model, tokenizer)

# Explain samples from the Reisinger dataset
reis_sample_patient = "Few telephone lines snapped ."
cls_explainer(reis_sample_patient + " " + "Few telephone lines was/were altered or somehow changed during or by the end of snapped.")
cls_explainer.visualize("vis_patient_altered.html")
cls_explainer(reis_sample_patient + " " + "The telephone lines was/were sentient.")
cls_explainer.visualize("vis_patient_sentient.html")
cls_explainer(reis_sample_patient + " " + "The telephone lines was/were used in carrying out snapped.")
cls_explainer.visualize("vis_patient_used.html")
cls_explainer(reis_sample_patient + " " + "The telephone lines was/were aware of being involved in snapped.")
cls_explainer.visualize("vis_patient_awareness.html")

reis_sample_agent = "He listens impassively ."
cls_explainer(reis_sample_agent + " " + "He was/were altered or somehow changed during or by the end of listens.")
cls_explainer.visualize("vis_agent_altered.html")
cls_explainer(reis_sample_agent + " " + "He was/were sentient.")
cls_explainer.visualize("vis_agent_sentient.html")
cls_explainer(reis_sample_agent + " " + "The telephone lines was/were used in carrying out snapped.")
cls_explainer.visualize("vis_agent_used.html")
cls_explainer(reis_sample_agent + " " + "The telephone lines was/were aware of being involved in snapped.")
cls_explainer.visualize("vis_agent_awareness.html")