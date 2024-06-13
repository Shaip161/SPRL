import json

# Ablation Test --> Remove from Dataset spr2 all entries where ispilot = true. --> these sentences are sentences that 
# work well with entalements questions.

# Load the JSON data from file
with open('datasets/json/spr2.json', 'r') as file:
    data = json.load(file)

# Filter out entries where "ispilot" is true
filtered_entries = {key: value for key, value in data.items() if not value.get('ispilot', False)}

# Write the filtered entries to a new file
with open('datasets/json/spr2_no_pilot.json', 'w') as file:
    json.dump(filtered_entries, file, indent=4)