import json
from nltk.corpus import propbank


def is_agent(roleset, arg_num):
    """
    This function determines if the given argument is an agent of given verb.
    The verb is defined by a PropBank roleset.

    Parameters:
        roleset (str): PropBank roleset of the verb.
        arg_num (int): Argument number (0 to 6).

    Returns:
        bool: True if the given argument is an agent, False otherwise. On input error, returns a fail string.
    """

    if roleset:
        lemma, set_num = roleset.split(".") # split the roleset into lemma and set number
        if not set_num.isdigit():
            print("Error: Invalid roleset number.")
            verb_roleset = propbank.roleset(f'{lemma}.01')
        else:
            verb_roleset = propbank.roleset(roleset)
            
        # Check if the argument number appears in roleset
        is_valid_arg = False
        for role in verb_roleset.findall("roles/role"):
            if role.attrib['n'] == str(arg_num):
                is_valid_arg = True
                break
        # If not, check all rolesets for presence of argument number
        if not is_valid_arg:
            for set in propbank.rolesets(lemma):
                for role in set.findall("roles/role"):
                    if role.attrib['n'] == str(arg_num):
                        is_valid_arg = True
                        verb_roleset = set
                        break
        # If no roleset contains the argument number, return fail value
        if not is_valid_arg:
            print("Error: Invalid argument number.")
            return "random invalid arg"

        # now we have roleset and argnum
        for role in verb_roleset.findall("roles/role"):
            if not role.findall("vnrole"):
                return "no vnrole"
            for vnrole in role.findall("vnrole"):
                if vnrole.attrib['vntheta'] in ['agent', 'Agent', 'actor1', 'Actor1'] :
                    return role.attrib['n'] == str(arg_num)
    
    return "random default"

with open("datasets/json/reis.json", "r") as f:
    reis_data = json.load(f)

for key, item in reis_data.items():
    label = is_agent(item['roleset'], item['arg_num'])
    if type(label) == bool:
        item['is_agent'] = int(label)

filtered_data = {key: item for key, item in reis_data.items() if 'is_agent' in item}

with open('datasets/json/reis_labelled.json', 'w') as f:
    json.dump(filtered_data, f, indent=4)