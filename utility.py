from collections import defaultdict
import pandas as pd

def generate_hypothesis(arg, pred, prop):
    """
    Generate a hypothesis based on the given arguments.

    Args:
        arg (str): The argument.
        pred (str): The predicate.
        prop (str): The property.

    Returns:
        str: The generated sentence.
    """
    layout_sentences = {
        'instigation': '{} caused {} to happen.',
        'volition': '{} chose to be involved in {}.',
        'awareness': '{} was/were aware of being involved in {}.',
        'sentient': '{} was/were sentient.',
        'change_of_location': '{} changed location during {}.',
        'exists_as_physical': '{} existed as a physical object.',
        'existed_before': '{} existed before {} began.',
        'existed_during': '{} existed during {}.',
        'existed_after': '{} existed after {} stopped.',
        'change_of_possession': '{} changed possession during {}.',
        'changes_possession': '{} changed possession during {}.',
        'change_of_state': '{} was/were altered or somehow changed during or by the end of {}.',
        'stationary': '{} was/were stationary during {}.',
        'location_of_event': '{} described the location of {}.',
        'makes_physical_contact': '{} made physical contact with someone or something else involved in {}.',
        'was_used': '{} was/were used in carrying out {}.',
        'manipulated_by_another': '{} was/were used in carrying out {}.',
        'predicate_changed_argument': 'The {} caused a change in {}.',
        'was_for_benefit': '{} happened for the benefit of {}.',
        'partitive': 'Only a part or portion of {} was involved in {}.',
        'change_of_state_continuous': 'The change in {} happened throughout {}.',
        'destroyed': '{} was/were destroyed during {}.',
        'created': '{} was/were created during {}.',
    }

    # Check if the property exists in layout sentences
    if prop in layout_sentences:
        sentence_template = layout_sentences[prop]
        sentence = sentence_template.format(arg, pred)
    else:
        print("Error: Prop was not found in layout_sentences.")
        return ""

    return sentence

def average_labels_over_entries(data):
    """
    Update each entry in the JSON data by averaging the labels for each property separately.

    Args:
        data: A dictionary containing JSON data where each key represents an entry and each value is another dictionary containing properties and labels.

    Returns:
        A modified dictionary where each entry has labels averaged over all occurrences of each property.
    """
    # Initialize a dictionary to store the modified data
    modified_data = {}

    # Iterate over each entry in the JSON data
    for entry_id, entry in data.items():
        # Initialize dictionaries to store the total labels and counts for each property
        property_label_totals = defaultdict(float)
        property_label_counts = defaultdict(int)
        applicability_counts = defaultdict(int)

        # Iterate over each property-label pair in the entry
        for prop, label, applic in zip(entry['property'], entry['label'], entry['applicable']):
            # Accumulate the label sum and count for each property
            property_label_counts[prop] += 1
            if applic:
                applicability_counts[prop] += 1
                property_label_totals[prop] += label
            else:
                applicability_counts[prop] += 0
                property_label_totals[prop] += 0

        # Initialize lists to store the averaged labels and corresponding properties
        averaged_labels = []
        averaged_properties = []
        averaged_applicability = []

        # Calculate the average label for each property in this entry
        for prop in property_label_totals:
            # If half or more of the entries are applicable, average over them
            if applicability_counts[prop] >= (property_label_counts[prop] / 2.0):
                averaged_label = property_label_totals[prop] / applicability_counts[prop]
                averaged_labels.append(averaged_label)
                averaged_properties.append(prop)
                averaged_applicability.append(True)
            # Else we denote this property as overall not applicable
            else:
                averaged_labels.append(-1.0)
                averaged_properties.append(prop)
                averaged_applicability.append(False)
            

        # Create a modified entry with the averaged labels and corresponding properties
        if 'ispilot' in entry:
            modified_entry = {
                'applicable': averaged_applicability,
                'property': averaged_properties,
                'ispilot': entry['ispilot'],
                'label': averaged_labels,
                'sentence': entry['sentence'],
                'split': entry['split'],
                'arg': entry['arg'],
                'pred': entry['pred']
            }
        elif 'is_agent' in entry:
            modified_entry = {
                'applicable': averaged_applicability,
                'property': averaged_properties,
                'label': averaged_labels,
                'sentence': entry['sentence'],
                'split': entry['split'],
                'arg': entry['arg'],
                'pred': entry['pred'],
                'is_agent': entry['is_agent']
            }
        else:
            modified_entry = {
                'applicable': averaged_applicability,
                'property': averaged_properties,
                'label': averaged_labels,
                'sentence': entry['sentence'],
                'split': entry['split'],
                'arg': entry['arg'],
                'pred': entry['pred']
            }

        # Add the modified entry to the modified data dictionary
        modified_data[entry_id] = modified_entry

    return modified_data

def read_json_data(data):
    """
    Reads JSON data and extracts premises, hypotheses, and labels for training, testing, and development sets.

    Args:
        data (dict): A dictionary containing JSON data with entries for training, testing, and development sets.

    Returns:
        tuple: A tuple containing pandas DataFrames for training, testing, and development sets.
               The tuple structure is as follows:
               (
                   train_df (DataFrame): DataFrame for the training set.
                   test_df (DataFrame): DataFrame for the testing set.
                   dev_df (DataFrame): DataFrame for the development set.
               )
    """
    # Initialize lists to store data
    train_data = []
    test_data = []
    dev_data = []

    # Iterate through JSON data
    for entry_id, entry in data.items():
        for prop, label, applic in zip(entry['property'], entry['label'], entry['applicable']):
            # Determine gold_label, entailement = 2, neutral = 1, contradiction = 0
            if applic:
                if label >= 4:
                    gold_label = 2
                else:
                    gold_label = 0
            else:
                gold_label = 1
            # Determine the split
            if entry['split'] == "train":
                train_data.append((entry['sentence'], generate_hypothesis(entry['arg'], entry['pred'], prop), gold_label))
            elif entry['split'] == "test":
                test_data.append((entry['sentence'], generate_hypothesis(entry['arg'], entry['pred'], prop), gold_label))
            elif entry['split'] == "dev":
                dev_data.append((entry['sentence'], generate_hypothesis(entry['arg'], entry['pred'], prop), gold_label))

    # Create DataFrames
    train_df = pd.DataFrame(train_data, columns=['premise', 'hypothesis', 'label'])
    test_df = pd.DataFrame(test_data, columns=['premise', 'hypothesis', 'label'])
    dev_df = pd.DataFrame(dev_data, columns=['premise', 'hypothesis', 'label'])

    return train_df, test_df, dev_df