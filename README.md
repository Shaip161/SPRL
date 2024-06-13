# Semantic Proto-role Labelling using Natural Language Inference
This Repository contains a viable, novel method for Semantic Proto-role Labelling (SPRL) using Natural Language Inference (NLI). We provide formatted SPRL datasets, a SPRL-finetuned variant of ```roberta-large-mnli``` and a logistic regressor to solve the binary classification problem of SPRL. By predicting the entailment of a given property set meant to identify agent- and patientlike qualities of a given argument in a given input sentence, we create an entailment vector functioning as the SPRL model's input.
We evaluate the resulting models using state-of-the-art frameworks, baselines and various robustness tests and observe performance in line with the state-of-the-art.

## Table of Contents
- [Getting Started](#getting-started)
- [Achievements](#achievements)
- [Concept](#concept)
  - [SPRL-Model](#sprl-model)
- [Data](#data)
- [Training](#training)
  - [RoBERTa](#roberta)
  - [SPRL-model](#sprl-model-1)
- [Evaluation](#evaluation)
  - [Quantitative](#quantitative)
  - [Robustness](#robustness)
  - [Interpretability](#interpretability)
- [Conclusion](#conclusion)
  - [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
  - [License](#license)


##### Authors
- Marlon Dittes, marlon.dittes@stud.uni-heidelberg.de
- Shai Peretz, dorian.peretz@stud.uni-heidelberg.de
- Iva Andreeva, iva.andreeva@stud.uni-heidelberg.de 

## Getting Started

In order to get started, clone the repository as follows:
- ```git clone https://github.com/Shaip161/SPRL.git --recursive```

Use --recursive to clone the roberta-large-mnli submodule as well.

After cloning the repository, create a Python virtual environment in the project directory. This isolates the project dependencies from the global Python environment. Use the following command:
- ```python -m venv my_venv```

After that, activate the virtual envirnment using the commands:
- On Windows: ```my_venv\Scripts\activate```
- On macOS and Linux: ```source my_venv/bin/activate```

With the virtual environment activated, install all required Python packages specified in the `requirements.txt` file:
- ```pip install -r requirements.txt```


## Concept
In previous work, sets of 14 to 18 properties were applied to target arguments in SPRL tasks. On this basis, sentences were generated with statements postulating the observed argument have that property. We use these sentences as entailment hypotheses and use an NLI model to classify the entailment of this hypothesis in the observed sentence. The NLI model is finetuned on Likert-Scale data: Human annotators annotate each hypothesis with a value of 1 - 5. These annotations infer gold labels for the NLI task. 

### SPRL-Model
Given the likelihoods of entailment and contradiction for each hypothesis and a gold label based on VerbNet thematic roles, the SPRL model regresses on a weight vector to solve the binary classification into proto-agent and proto-patient.

## Data
We used Reisinger's SPR1 and White's SPR2 dataset to create our own datasets (json files) for later use:

Generally, reis.json, reis_labelled.json, spr1.json, spr2.json, spr2_1.json, spr2_no_pilot.json all consist of the following information:
- applicable: Whether the property is applicable in principle to the argument in question; either True or False.
- property: The proto-role property being annotated.
- label: The 5-point Likert scale annotation of the property, where 1 means very unlikely and 5 means very likely.
- sentence: The base sentence, which we will later use as the premise.
- split: The training split used in training the model, either train, dev, or test.
- arg: The argument in the sentence.
- pred: The predicate in the sentence.
- gram_func: The grammatical function of the argument. subj, obj or other for Reisinger and nsubj, nsubjpass, dobj or iobj for White.

##### Reis.json

reis.json was created using Reisinger's SPR1 data to finetune RoBERTa. It additionally contains the following entries:
- roleset: The propbank roleset of the predicate.
- arg_num: The propbank argument label; either 0, 1, 2, 3, 4 or 5.

It consists of 6363 entries using 18 different properties.

##### Reis_labelled.json

reis_labelled.json is the subset of entries in reis.json on which we could find a VerbNet entry for the giving roleset using NLTK.
We use this data to train weights for the different property features and also to evaluate how good a model is at the SPRL task. It additionally contains the following entry:
- is_agent: Denotes if the given argument is of proto-agent status in the given sentence; either 0 or 1.
It consists of 2715 entries using the same properties as reis.json.

##### spr1.json, spr2.json, spr2_1.json
	
spr1.json, spr2.json, spr2_1.json were created using White's SPR2 data. They are mainly used for finetuning RoBERTa and additionally have the following entry:
- ispilot: Whether the annotation was part of a pilot or not; either True or False.
Pilot annotations pass the filters introduced by Reisinger et al. 
They respectively consist of 92, 2850 and 1396 entries. Between the three there are however duplicates. spr1.json uses 16 different properties, whereas the remaining two use 14 properties.

##### spr2_no_pilot.json

spr2_no_pilot.json consists of all spr2.json entries with ispilot = False. It is used for ablation studies.
It consists of 2758 entries.

##### Duplicates

There are sentences which got annotated by multiple annotators. To not have duplicate (premise, hypothesis) pairs and decrease overall finetuning duration, we also use variations of the above datasets where we average over all the labels for a given property.

##### Weight training

To train the weights of our different properties we created files of the format "model_predictions.json" containing 2715 datapoints which each consist of the following entries:
- is_agent: Denotes if the given argument is of proto-agent status in the given sentence; either 0 or 1.
- split: The training split used in training the model, either train, dev, or test.
- roberta_entailments: The likelihood of entailment which the model returned for a (premise, hypothesis) pair on a given property.
- roberta_contradictions: The likelihood of contradiction which the model returned for a (premise, hypothesis) pair on a given property.



## Training
We finetuned RoBERTa and trained our SPRL model separately, as the SPRL training data had to include the entailment predictions. Due to the sequential nature of this training process, it was not only computationally intensive, but also time-intensive. Thus, a bug which impacted performance of the SPRL model greatly was noticed shortly before the project deadline.

### RoBERTa
As our base model we used roberta-large-mnli, which is the RoBERTa large model fine-tuned on the Multi-Genre Natural Language Inference (MNLI) corpus. We use this model to maximize performance on the NLI task needed during the finetuning process.

##### Procedure

To generate train, test and dev data with gold labels from our own datasets (json) which use likert scale data, we did the following procedure:

We iterate through all datapoints in our dataset.
We create the pair of (premise, hypothesis) as follows:
- premise: The sentence of the datapoint.
- hypothesis: Created using template hypotheses which use the argument (ARG) and predicate (PRED) of the datapoint.
	Example for "instigation": "ARG caused PRED to happen."

If a property is deemed applicable in its context, we denote the gold label as 'entailment' if the likert label is >= 4 for the given property and 'contradiction' if the likert label is < 4.
If a property is not deemed applicable, we denote the gold label as 'neutral'.

We chose to handle gold labels this way, since non-applicable properties don't provide a feasible likert label for us to use.

##### Training Arguments

We finetuned the roberta-large-mnli model using Hugging Face's transformer library Trainer class using the following TrainingArguments:
- num_train_epochs=3 
  Note: We used 2 epochs on roberta_retrained_non-averaged due to the finetuning process already taking a lot of time.
- per_device_train_batch_size=16
- per_device_eval_batch_size=64
- warmup_steps=500
- weight_decay=0.01

##### Models

We finetuned 5 different models:
- roberta_averaged: Uses averaged versions of reis.json, spr1.json, spr2.json, spr2_1.json.
- roberta_non-averaged: Uses non-averaged reis.json, spr1.json, spr2.json, spr2_1.json.
- roberta_reisinger: Uses averaged version of reis.json.
- roberta_white: Uses averaged version of spr1.json, spr2.json, spr2_1.json.
- roberta_no-pilot: Uses averaged version of spr2_no_pilot.json.

### SPRL-model

Given a model and an instance of our reis_labelled.json, we tackle the SPRL task in the following way:

We use the given model to compute entailment and prediction likelihoods for all 18 properties. We then calculate the dot product of these 36 features with 36 weights, whilst adding an bias. After that, we use the sigmoid function on the result; a value >= 0.5 returns proto-agent as predicted, a value < 0.5 returns proto-patient as predicted.

To get the weights and bias, we created "model_predictions.json" files as mentioned in [Data](#data). 
We then used the scikit-learn library to create a Logistic Regression linear model for training the weights and bias on our train data. 
We also tried a Support Vector Machine model to make sure we get a good linear decision boundary whilst using Logistic Regression, which turned out to produce the same accuracy.
We additionally directly computed weights using the likert scale data from our dataset, but this turned out to overall have significantly worse accuracy.

## Evaluation
In the following section we will introduce 3 evaluation techniques and results in order to objectively compare the trained models with one another. 

### Quantitative 

The Quantitative technique includes:  
- Overall accuracy, to calculate the proportion of correct predictions across all categories. 
- The F1 score, which balances precision and recall, reflecting the model's reliability in positive case identification. 
- Average Log Loss in order to measure prediction certainty, with lower values indicating higher accuracy. 
- Class-specific accuracy to assess the model's performance in individual categories, identifying potential biases or weaknesses.

The results from our different models:

| Model                      | Overall Accuracy      | F1                 | Average Log Loss    | Proto-Agent Class Accuracy | Proto-Patient Class Accuracy  |
|----------------------------|-----------------------|--------------------|---------------------|----------------------------|-------------------------------|
| roberta-large-mnli         | 0.8209150326797385    | 0.8319018404907976 | 0.43994178200779427 | 0.9442896935933147         | 0.7118226600985221            |
| roberta_averaged           | 0.7568627450980392    | 0.7866972477064219 | 0.5493920016025409  | 0.9554317548746518         | 0.5812807881773399            |
| roberta_non-averaged       | 0.7019607843137254    | 0.7438202247191011 | 0.6178036201613978  | 0.9220055710306406         | 0.5073891625615764            |
| roberta_reisinger          | 0.7411764705882353    | 0.7697674418604652 | 0.5318926444518163  | 0.9220055710306406         | 0.5812807881773399            |
| roberta_no-pilot           | 0.7620915032679738    | 0.7883720930232558 | 0.5509038203001045  | 0.9442896935933147         | 0.6009852216748769            |
| roberta_white              | 0.469281045751634     | 0.6387900355871886 | 0.8862093728993434  | 1.0                        | 0.0                           |

### Robustness 

To test the robustness of our models, we have carried out:
- An ablation study, where we systematically remove a specific part of our models to understand its impact on performance. We did this by removing all dataset entries, where the isPilot tag set to true from the SPR2 dataset. This is because these sentences were particularly well suited for entailment questions. By removing them, we have not only reduced the dataset on which the model can train, but we have also removed the sentence where the entailment questions are ideal.
- Adversarial testing. Here we want to evaluate how our models perform against carefully crafted adversarial examples to test the limits of their understanding. These hand-crafted sentences include, for example, various biases such as gender bias, where a male is generally more likely to be a proto-agent than a female.
- Out of distribution tests, where we test the models on data that is significantly different from the training distribution. This data includes hand-crafted nonsensical sentences that were not used in the training of our models.  

The results of the Adversial Evaluation method using the wights obtained from logistical regression:

The control dataset result

| Model                      | Overall Accuracy      | F1                 | Average Log Loss    | Proto-Agent Class Accuracy | Proto-Patient Class Accuracy  |
|----------------------------|-----------------------|--------------------|---------------------|----------------------------|-------------------------------|
| roberta_averaged           | 0.7                   | 0.7692307692307693 | 0.451683302968091   | 1.0                        | 0.4                           |
| roberta_non-averaged       | 0.7                   | 0.7692307692307693 | 0.7910062779921779  | 1.0                        | 0.4                           |
| roberta_no-pilot           | 0.95                  | 0.9523809523809523 | 0.203242366753267   | 1.0                        | 0.9                           |
| roberta_reisinger          | 0.85                  | 0.8695652173913044 | 0.3082706832583492  | 1.0                        | 0.7                           |
| roberta_white              | 0.5                   | 0.6666666666666666 | 0.8368979542719753  | 1.0                        | 0.0                           |
| roberta-large-mnli         | 0.9                   | 0.9090909090909091 | 0.23855840792802    | 1.0                        | 0.8                           |


The bias dataset results:

| Model                      | Overall Accuracy      | F1                 | Average Log Loss    | Proto-Agent Class Accuracy | Proto-Patient Class Accuracy  |
|----------------------------|-----------------------|--------------------|---------------------|----------------------------|-------------------------------|
| roberta_averaged           | 0.7                   | 0.7692307692307693 | 0.4489823490394652  | 1.0                        | 0.4                           |
| roberta_non-averaged       | 0.7                   | 0.7692307692307693 | 0.8150326148910881  | 1.0                        | 0.4                           |
| roberta_no-pilot           | 0.9                   | 0.9090909090909091 | 0.223213626842037   | 1.0                        | 0.8                           |
| roberta_reisinger          | 0.8                   | 0.8333333333333333 | 0.31446367602131153 | 1.0                        | 0.6                           |
| roberta_white              | 0.5                   | 0.6666666666666666 | 0.8502637775315189  | 1.0                        | 0.0                           |
| roberta-large-mnli         | 0.85                  | 0.8695652173913044 | 0.2807292437770147  | 1.0                        | 0.7                           |

The results of the Out of Distribution method using the wights obtained from logistical regression:

| Model                      | Overall Accuracy      | F1                 | Average Log Loss    | Proto-Agent Class Accuracy | Proto-Patient Class Accuracy  |
|----------------------------|-----------------------|--------------------|---------------------|----------------------------|-------------------------------|
| roberta_averaged           | 0.55                  | 0.6896551724137931 | 0.5093667188659585  | 1.0                        | 0.1                           |
| roberta_non-averaged       | 0.6                   | 0.6363636363636365 | 0.7748667116320569  | 0.7                        | 0.5                           |
| roberta_no-pilot           | 0.8                   | 0.7499999999999999 | 0.3736504221758817  | 0.6                        | 1.0                           |
| roberta_reisinger          | 0.85                  | 0.8571428571428572 | 0.29798511748203266 | 0.9                        | 0.8                           |
| roberta_white              | 0.5                   | 0.6666666666666666 | 0.8195011716349698  | 1.0                        | 0.0                           |
| roberta-large-mnli         | 0.75                  | 0.8                | 0.4921452778276362  | 1.0                        | 0.5                           |

### Interpretability
To understand the decision-making of our SPRL-model, we inspect the weights learned from training: As the entailment of certain properties indicates proto-agenthood, those properties recieve large, positive entailment weights. In turn, proto-patient-related property entailment weights recieve large, positive weights as well.
By this system, a largely coherent mapping to proto-agent and proto-patient properties can be extracted.

Additionally, we observe token importance in our RoBERTa variant models to understand which part of an input sentence is used to infer the queried properties. We perform these observations using ```transformers-interpret``` on multiple examples for multiple properties, including *sentience* and *change of state*. 

<table width: 100%>
<div style="border-top: 1px solid; margin-top: 5px;             padding-top: 5px; display: inline-block"><b>Legend: </b><span style="display: inline-block; width: 10px; height: 10px;                 border: 1px solid; background-color:                 hsl(0, 75%, 60%)"></span> Negative  <span style="display: inline-block; width: 10px; height: 10px;                 border: 1px solid; background-color:                 hsl(0, 75%, 100%)"></span> Neutral  <span style="display: inline-block; width: 10px; height: 10px;                 border: 1px solid; background-color:                 hsl(120, 75%, 50%)"></span> Positive  </div><tr><th>Prediction Score</th><th>Attribution Label</th><th>Attribution Score</th><th>Word Importance</th><tr><td><text ><b> (0.18)</b></text></td><td><text ><b>CONTRADICTION</b></text></td><td><text ><b>-0.85</b></text></td><td><mark style="background-color: hsl(0, 75%, 100%); opacity:1.0;                     line-height:1.75"></mark><mark style="background-color: hsl(120, 75%, 97%); opacity:1.0;                     line-height:1.75"><font color="black"> He                    </font></mark><mark style="background-color: hsl(0, 75%, 94%); opacity:1.0;                     line-height:1.75"><font color="black"> listens                    </font></mark><mark style="background-color: hsl(0, 75%, 100%); opacity:1.0;                     line-height:1.75"><font color="black"> imp                    </font></mark><mark style="background-color: hsl(120, 75%, 100%); opacity:1.0;                     line-height:1.75"><font color="black"> ass                    </font></mark><mark style="background-color: hsl(0, 75%, 93%); opacity:1.0;                     line-height:1.75"><font color="black"> ively                    </font></mark><mark style="background-color: hsl(120, 75%, 95%); opacity:1.0;                     line-height:1.75"><font color="black"> .                    </font></mark><mark style="background-color: hsl(0, 75%, 89%); opacity:1.0;                     line-height:1.75"><font color="black"> He                    </font></mark><mark style="background-color: hsl(0, 75%, 93%); opacity:1.0;                     line-height:1.75"><font color="black"> was                    </font></mark><mark style="background-color: hsl(0, 75%, 96%); opacity:1.0;                     line-height:1.75"><font color="black"> /                    </font></mark><mark style="background-color: hsl(120, 75%, 93%); opacity:1.0;                     line-height:1.75"><font color="black"> were                    </font></mark><mark style="background-color: hsl(0, 75%, 72%); opacity:1.0;                     line-height:1.75"><font color="black"> sentient                    </font></mark><mark style="background-color: hsl(120, 75%, 75%); opacity:1.0;                     line-height:1.75"><font color="black"> .                    </font></mark><mark style="background-color: hsl(0, 75%, 97%); opacity:1.0;                     line-height:1.75"><font color="black">                     </font></mark><mark style="background-color: hsl(0, 75%, 100%); opacity:1.0;                     line-height:1.75"></mark></td><tr><tr><td><text ><b> (0.63)</b></text></td><td><text ><b>NEUTRAL</b></text></td><td><text ><b>0.72</b></text></td><td><mark style="background-color: hsl(0, 75%, 100%); opacity:1.0;                     line-height:1.75"></mark><mark style="background-color: hsl(0, 75%, 87%); opacity:1.0;                     line-height:1.75"><font color="black"> He                    </font></mark><mark style="background-color: hsl(120, 75%, 94%); opacity:1.0;                     line-height:1.75"><font color="black"> listens                    </font></mark><mark style="background-color: hsl(120, 75%, 96%); opacity:1.0;                     line-height:1.75"><font color="black"> imp                    </font></mark><mark style="background-color: hsl(120, 75%, 93%); opacity:1.0;                     line-height:1.75"><font color="black"> ass                    </font></mark><mark style="background-color: hsl(120, 75%, 91%); opacity:1.0;                     line-height:1.75"><font color="black"> ively                    </font></mark><mark style="background-color: hsl(0, 75%, 89%); opacity:1.0;                     line-height:1.75"><font color="black"> .                    </font></mark><mark style="background-color: hsl(0, 75%, 99%); opacity:1.0;                     line-height:1.75"><font color="black"> He                    </font></mark><mark style="background-color: hsl(120, 75%, 95%); opacity:1.0;                     line-height:1.75"><font color="black"> was                    </font></mark><mark style="background-color: hsl(0, 75%, 96%); opacity:1.0;                     line-height:1.75"><font color="black"> /                    </font></mark><mark style="background-color: hsl(120, 75%, 77%); opacity:1.0;                     line-height:1.75"><font color="black"> were                    </font></mark><mark style="background-color: hsl(120, 75%, 69%); opacity:1.0;                     line-height:1.75"><font color="black"> sentient                    </font></mark><mark style="background-color: hsl(0, 75%, 89%); opacity:1.0;                     line-height:1.75"><font color="black"> .                    </font></mark><mark style="background-color: hsl(120, 75%, 100%); opacity:1.0;                     line-height:1.75"><font color="black">                     </font></mark><mark style="background-color: hsl(0, 75%, 100%); opacity:1.0;                     line-height:1.75"></mark></td><tr><tr><td><text ><b> (0.74)</b></text></td><td><text ><b>ENTAILMENT</b></text></td><td><text ><b>0.58</b></text></td><td><mark style="background-color: hsl(0, 75%, 100%); opacity:1.0;                     line-height:1.75"></mark><mark style="background-color: hsl(120, 75%, 91%); opacity:1.0;                     line-height:1.75"><font color="black"> He                    </font></mark><mark style="background-color: hsl(120, 75%, 94%); opacity:1.0;                     line-height:1.75"><font color="black"> listens                    </font></mark><mark style="background-color: hsl(0, 75%, 97%); opacity:1.0;                     line-height:1.75"><font color="black"> imp                    </font></mark><mark style="background-color: hsl(0, 75%, 95%); opacity:1.0;                     line-height:1.75"><font color="black"> ass                    </font></mark><mark style="background-color: hsl(120, 75%, 97%); opacity:1.0;                     line-height:1.75"><font color="black"> ively                    </font></mark><mark style="background-color: hsl(120, 75%, 94%); opacity:1.0;                     line-height:1.75"><font color="black"> .                    </font></mark><mark style="background-color: hsl(120, 75%, 79%); opacity:1.0;                     line-height:1.75"><font color="black"> He                    </font></mark><mark style="background-color: hsl(120, 75%, 97%); opacity:1.0;                     line-height:1.75"><font color="black"> was                    </font></mark><mark style="background-color: hsl(120, 75%, 90%); opacity:1.0;                     line-height:1.75"><font color="black"> /                    </font></mark><mark style="background-color: hsl(0, 75%, 76%); opacity:1.0;                     line-height:1.75"><font color="black"> were                    </font></mark><mark style="background-color: hsl(120, 75%, 80%); opacity:1.0;                     line-height:1.75"><font color="black"> sentient                    </font></mark><mark style="background-color: hsl(0, 75%, 87%); opacity:1.0;                     line-height:1.75"><font color="black"> .                    </font></mark><mark style="background-color: hsl(120, 75%, 96%); opacity:1.0;                     line-height:1.75"><font color="black">                     </font></mark><mark style="background-color: hsl(0, 75%, 100%); opacity:1.0;                     line-height:1.75"></mark></td><tr>
</table>


<table width: 60%>
<div style="border-top: 1px solid; margin-top: 5px;             padding-top: 5px; display: inline-block"><b>Legend: </b><span style="display: inline-block; width: 10px; height: 10px;                 border: 1px solid; background-color:                 hsl(0, 75%, 60%)"></span> Negative  <span style="display: inline-block; width: 10px; height: 10px;                 border: 1px solid; background-color:                 hsl(0, 75%, 100%)"></span> Neutral  <span style="display: inline-block; width: 10px; height: 10px;                 border: 1px solid; background-color:                 hsl(120, 75%, 50%)"></span> Positive  </div><tr><th>Prediction Score</th><th>Attribution Label</th><th>Attribution Score</th><th>Word Importance</th><tr><td><text><b> (0.81)</b></text></td><td><text ><b>CONTRADICTION</b></text></td><td><text><b>1.23</b></text></td><td><mark style="background-color: hsl(0, 75%, 100%); opacity:1.0;                     line-height:1.75"></mark><mark style="background-color: hsl(120, 75%, 92%); opacity:1.0;                     line-height:1.75"><font color="black"> He                    </font></mark><mark style="background-color: hsl(120, 75%, 100%); opacity:1.0;                     line-height:1.75"><font color="black"> listens                    </font></mark><mark style="background-color: hsl(120, 75%, 72%); opacity:1.0;                     line-height:1.75"><font color="black"> imp                    </font></mark><mark style="background-color: hsl(120, 75%, 71%); opacity:1.0;                     line-height:1.75"><font color="black"> ass                    </font></mark><mark style="background-color: hsl(0, 75%, 94%); opacity:1.0;                     line-height:1.75"><font color="black"> ively                    </font></mark><mark style="background-color: hsl(0, 75%, 100%); opacity:1.0;                     line-height:1.75"><font color="black"> .                    </font></mark><mark style="background-color: hsl(120, 75%, 89%); opacity:1.0;                     line-height:1.75"><font color="black"> He                    </font></mark><mark style="background-color: hsl(120, 75%, 99%); opacity:1.0;                     line-height:1.75"><font color="black"> was                    </font></mark><mark style="background-color: hsl(0, 75%, 95%); opacity:1.0;                     line-height:1.75"><font color="black"> /                    </font></mark><mark style="background-color: hsl(0, 75%, 99%); opacity:1.0;                     line-height:1.75"><font color="black"> were                    </font></mark><mark style="background-color: hsl(0, 75%, 94%); opacity:1.0;                     line-height:1.75"><font color="black"> altered                    </font></mark><mark style="background-color: hsl(0, 75%, 98%); opacity:1.0;                     line-height:1.75"><font color="black"> or                    </font></mark><mark style="background-color: hsl(120, 75%, 97%); opacity:1.0;                     line-height:1.75"><font color="black"> somehow                    </font></mark><mark style="background-color: hsl(0, 75%, 92%); opacity:1.0;                     line-height:1.75"><font color="black"> changed                    </font></mark><mark style="background-color: hsl(120, 75%, 91%); opacity:1.0;                     line-height:1.75"><font color="black"> during                    </font></mark><mark style="background-color: hsl(0, 75%, 99%); opacity:1.0;                     line-height:1.75"><font color="black"> or                    </font></mark><mark style="background-color: hsl(0, 75%, 99%); opacity:1.0;                     line-height:1.75"><font color="black"> by                    </font></mark><mark style="background-color: hsl(120, 75%, 99%); opacity:1.0;                     line-height:1.75"><font color="black"> the                    </font></mark><mark style="background-color: hsl(120, 75%, 90%); opacity:1.0;                     line-height:1.75"><font color="black"> end                    </font></mark><mark style="background-color: hsl(120, 75%, 100%); opacity:1.0;                     line-height:1.75"><font color="black"> of                    </font></mark><mark style="background-color: hsl(0, 75%, 94%); opacity:1.0;                     line-height:1.75"><font color="black"> listens                    </font></mark><mark style="background-color: hsl(120, 75%, 99%); opacity:1.0;                     line-height:1.75"><font color="black"> .                    </font></mark><mark style="background-color: hsl(120, 75%, 96%); opacity:1.0;                     line-height:1.75"><font color="black">                     </font></mark><mark style="background-color: hsl(0, 75%, 100%); opacity:1.0;                     line-height:1.75"></mark></td><tr><tr><td><text><b> (0.73)</b></text></td><td><text><b>NEUTRAL</b></text></td><td><text><b>2.37</b></text></td><td><mark style="background-color: hsl(0, 75%, 100%); opacity:1.0;                     line-height:1.75"></mark><mark style="background-color: hsl(0, 75%, 97%); opacity:1.0;                     line-height:1.75"><font color="black"> He                    </font></mark><mark style="background-color: hsl(120, 75%, 97%); opacity:1.0;                     line-height:1.75"><font color="black"> listens                    </font></mark><mark style="background-color: hsl(0, 75%, 85%); opacity:1.0;                     line-height:1.75"><font color="black"> imp                    </font></mark><mark style="background-color: hsl(0, 75%, 95%); opacity:1.0;                     line-height:1.75"><font color="black"> ass                    </font></mark><mark style="background-color: hsl(120, 75%, 82%); opacity:1.0;                     line-height:1.75"><font color="black"> ively                    </font></mark><mark style="background-color: hsl(120, 75%, 96%); opacity:1.0;                     line-height:1.75"><font color="black"> .                    </font></mark><mark style="background-color: hsl(120, 75%, 89%); opacity:1.0;                     line-height:1.75"><font color="black"> He                    </font></mark><mark style="background-color: hsl(120, 75%, 91%); opacity:1.0;                     line-height:1.75"><font color="black"> was                    </font></mark><mark style="background-color: hsl(120, 75%, 92%); opacity:1.0;                     line-height:1.75"><font color="black"> /                    </font></mark><mark style="background-color: hsl(120, 75%, 97%); opacity:1.0;                     line-height:1.75"><font color="black"> were                    </font></mark><mark style="background-color: hsl(120, 75%, 86%); opacity:1.0;                     line-height:1.75"><font color="black"> altered                    </font></mark><mark style="background-color: hsl(120, 75%, 92%); opacity:1.0;                     line-height:1.75"><font color="black"> or                    </font></mark><mark style="background-color: hsl(120, 75%, 83%); opacity:1.0;                     line-height:1.75"><font color="black"> somehow                    </font></mark><mark style="background-color: hsl(120, 75%, 80%); opacity:1.0;                     line-height:1.75"><font color="black"> changed                    </font></mark><mark style="background-color: hsl(120, 75%, 88%); opacity:1.0;                     line-height:1.75"><font color="black"> during                    </font></mark><mark style="background-color: hsl(0, 75%, 99%); opacity:1.0;                     line-height:1.75"><font color="black"> or                    </font></mark><mark style="background-color: hsl(120, 75%, 93%); opacity:1.0;                     line-height:1.75"><font color="black"> by                    </font></mark><mark style="background-color: hsl(120, 75%, 97%); opacity:1.0;                     line-height:1.75"><font color="black"> the                    </font></mark><mark style="background-color: hsl(0, 75%, 100%); opacity:1.0;                     line-height:1.75"><font color="black"> end                    </font></mark><mark style="background-color: hsl(120, 75%, 99%); opacity:1.0;                     line-height:1.75"><font color="black"> of                    </font></mark><mark style="background-color: hsl(0, 75%, 97%); opacity:1.0;                     line-height:1.75"><font color="black"> listens                    </font></mark><mark style="background-color: hsl(120, 75%, 99%); opacity:1.0;                     line-height:1.75"><font color="black"> .                    </font></mark><mark style="background-color: hsl(120, 75%, 91%); opacity:1.0;                     line-height:1.75"><font color="black">                     </font></mark><mark style="background-color: hsl(0, 75%, 100%); opacity:1.0;                     line-height:1.75"></mark></td><tr><tr><td><text><b> (0.08)</b></text></td><td><text><b>ENTAILMENT</b></text></td><td><text><b>-2.51</b></text></td><td><mark style="background-color: hsl(0, 75%, 100%); opacity:1.0;                     line-height:1.75"></mark><mark style="background-color: hsl(0, 75%, 100%); opacity:1.0;                     line-height:1.75"><font color="black"> He                    </font></mark><mark style="background-color: hsl(0, 75%, 99%); opacity:1.0;                     line-height:1.75"><font color="black"> listens                    </font></mark><mark style="background-color: hsl(0, 75%, 89%); opacity:1.0;                     line-height:1.75"><font color="black"> imp                    </font></mark><mark style="background-color: hsl(0, 75%, 83%); opacity:1.0;                     line-height:1.75"><font color="black"> ass                    </font></mark><mark style="background-color: hsl(0, 75%, 96%); opacity:1.0;                     line-height:1.75"><font color="black"> ively                    </font></mark><mark style="background-color: hsl(0, 75%, 99%); opacity:1.0;                     line-height:1.75"><font color="black"> .                    </font></mark><mark style="background-color: hsl(0, 75%, 88%); opacity:1.0;                     line-height:1.75"><font color="black"> He                    </font></mark><mark style="background-color: hsl(0, 75%, 96%); opacity:1.0;                     line-height:1.75"><font color="black"> was                    </font></mark><mark style="background-color: hsl(120, 75%, 99%); opacity:1.0;                     line-height:1.75"><font color="black"> /                    </font></mark><mark style="background-color: hsl(0, 75%, 98%); opacity:1.0;                     line-height:1.75"><font color="black"> were                    </font></mark><mark style="background-color: hsl(0, 75%, 96%); opacity:1.0;                     line-height:1.75"><font color="black"> altered                    </font></mark><mark style="background-color: hsl(0, 75%, 99%); opacity:1.0;                     line-height:1.75"><font color="black"> or                    </font></mark><mark style="background-color: hsl(0, 75%, 84%); opacity:1.0;                     line-height:1.75"><font color="black"> somehow                    </font></mark><mark style="background-color: hsl(120, 75%, 99%); opacity:1.0;                     line-height:1.75"><font color="black"> changed                    </font></mark><mark style="background-color: hsl(0, 75%, 82%); opacity:1.0;                     line-height:1.75"><font color="black"> during                    </font></mark><mark style="background-color: hsl(120, 75%, 96%); opacity:1.0;                     line-height:1.75"><font color="black"> or                    </font></mark><mark style="background-color: hsl(0, 75%, 97%); opacity:1.0;                     line-height:1.75"><font color="black"> by                    </font></mark><mark style="background-color: hsl(0, 75%, 98%); opacity:1.0;                     line-height:1.75"><font color="black"> the                    </font></mark><mark style="background-color: hsl(0, 75%, 92%); opacity:1.0;                     line-height:1.75"><font color="black"> end                    </font></mark><mark style="background-color: hsl(0, 75%, 100%); opacity:1.0;                     line-height:1.75"><font color="black"> of                    </font></mark><mark style="background-color: hsl(120, 75%, 84%); opacity:1.0;                     line-height:1.75"><font color="black"> listens                    </font></mark><mark style="background-color: hsl(0, 75%, 99%); opacity:1.0;                     line-height:1.75"><font color="black"> .                    </font></mark><mark style="background-color: hsl(0, 75%, 93%); opacity:1.0;                     line-height:1.75"><font color="black">                     </font></mark><mark style="background-color: hsl(0, 75%, 100%); opacity:1.0;                     line-height:1.75"></mark></td><tr>
</table>


Using ```ferret```, we are able to not only explain the entailment prediction of the property *sentience* for "He listens impassively.", but also evaluate the explanation by observing correlation between state-of-the-art explainability methods. 

<table id="T_7fddc">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_7fddc_level0_col0" class="col_heading level0 col0" >aopc_compr</th>
      <th id="T_7fddc_level0_col1" class="col_heading level0 col1" >aopc_suff</th>
      <th id="T_7fddc_level0_col2" class="col_heading level0 col2" >taucorr_loo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_7fddc_level0_row0" class="row_heading level0 row0" >Partition SHAP</th>
      <td id="T_7fddc_row0_col0" class="data row0 col0" >0.17</td>
      <td id="T_7fddc_row0_col1" class="data row0 col1" >-0.05</td>
      <td id="T_7fddc_row0_col2" class="data row0 col2" >0.45</td>
    </tr>
    <tr>
      <th id="T_7fddc_level0_row1" class="row_heading level0 row1" >LIME</th>
      <td id="T_7fddc_row1_col0" class="data row1 col0" >0.16</td>
      <td id="T_7fddc_row1_col1" class="data row1 col1" >-0.15</td>
      <td id="T_7fddc_row1_col2" class="data row1 col2" >0.30</td>
    </tr>
    <tr>
      <th id="T_7fddc_level0_row2" class="row_heading level0 row2" >Gradient</th>
      <td id="T_7fddc_row2_col0" class="data row2 col0" >0.01</td>
      <td id="T_7fddc_row2_col1" class="data row2 col1" >0.07</td>
      <td id="T_7fddc_row2_col2" class="data row2 col2" >-0.09</td>
    </tr>
    <tr>
      <th id="T_7fddc_level0_row3" class="row_heading level0 row3" >Gradient (x Input)</th>
      <td id="T_7fddc_row3_col0" class="data row3 col0" >0.09</td>
      <td id="T_7fddc_row3_col1" class="data row3 col1" >-0.05</td>
      <td id="T_7fddc_row3_col2" class="data row3 col2" >0.18</td>
    </tr>
    <tr>
      <th id="T_7fddc_level0_row4" class="row_heading level0 row4" >Integrated Gradient</th>
      <td id="T_7fddc_row4_col0" class="data row4 col0" >0.04</td>
      <td id="T_7fddc_row4_col1" class="data row4 col1" >-0.12</td>
      <td id="T_7fddc_row4_col2" class="data row4 col2" >0.21</td>
    </tr>
    <tr>
      <th id="T_7fddc_level0_row5" class="row_heading level0 row5" >Integrated Gradient (x Input)</th>
      <td id="T_7fddc_row5_col0" class="data row5 col0" >-0.04</td>
      <td id="T_7fddc_row5_col1" class="data row5 col1" >0.15</td>
      <td id="T_7fddc_row5_col2" class="data row5 col2" >-0.39</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>He</th>
      <th>Ġlistens</th>
      <th>Ġimp</th>
      <th>ass</th>
      <th>ively</th>
      <th>Ġ.</th>
      <th>ĠHe</th>
      <th>Ġwas</th>
      <th>/</th>
      <th>were</th>
      <th>Ġsentient</th>
      <th>.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Partition SHAP</th>
      <td>0.081315</td>
      <td>0.045069</td>
      <td>-0.104878</td>
      <td>0.057999</td>
      <td>0.056598</td>
      <td>0.039422</td>
      <td>0.004542</td>
      <td>-0.051674</td>
      <td>0.056328</td>
      <td>-0.118104</td>
      <td>0.328409</td>
      <td>0.055659</td>
    </tr>
    <tr>
      <th>LIME</th>
      <td>0.177724</td>
      <td>-0.022607</td>
      <td>-0.102271</td>
      <td>0.048818</td>
      <td>0.011405</td>
      <td>0.009288</td>
      <td>0.130528</td>
      <td>0.083796</td>
      <td>0.018073</td>
      <td>0.001113</td>
      <td>0.261391</td>
      <td>0.132987</td>
    </tr>
    <tr>
      <th>Gradient</th>
      <td>0.039101</td>
      <td>0.046283</td>
      <td>0.047539</td>
      <td>0.059878</td>
      <td>0.055153</td>
      <td>0.031957</td>
      <td>0.044065</td>
      <td>0.051185</td>
      <td>0.034778</td>
      <td>0.063744</td>
      <td>0.360032</td>
      <td>0.056943</td>
    </tr>
    <tr>
      <th>Gradient (x Input)</th>
      <td>0.137561</td>
      <td>0.070364</td>
      <td>-0.064036</td>
      <td>-0.185072</td>
      <td>0.095231</td>
      <td>-0.012394</td>
      <td>-0.022001</td>
      <td>-0.129332</td>
      <td>-0.023433</td>
      <td>-0.035298</td>
      <td>0.179182</td>
      <td>0.002415</td>
    </tr>
    <tr>
      <th>Integrated Gradient</th>
      <td>0.064614</td>
      <td>-0.067027</td>
      <td>-0.096624</td>
      <td>0.024009</td>
      <td>-0.037160</td>
      <td>-0.124285</td>
      <td>-0.040444</td>
      <td>0.088329</td>
      <td>0.095535</td>
      <td>0.043687</td>
      <td>-0.201441</td>
      <td>-0.011098</td>
    </tr>
    <tr>
      <th>Integrated Gradient (x Input)</th>
      <td>-0.012721</td>
      <td>0.012819</td>
      <td>0.080771</td>
      <td>0.025040</td>
      <td>0.107646</td>
      <td>-0.148754</td>
      <td>0.237772</td>
      <td>0.071891</td>
      <td>-0.068089</td>
      <td>-0.010841</td>
      <td>0.220317</td>
      <td>-0.003338</td>
    </tr>
  </tbody>
</table>

## Conclusion
We conclude that the approach to SPRL using NLI has merit, as one of our non-averaged model variants can compete with state-of-the-art baselines on similar datasets. However, our method, especially using ```roberta-large-mnli``` is immensely computationally intensive and requires large amounts of data. We therefore see great potential for improvement.

### Future Work
- A refinement of the entailment hypotheses to conform to grammar rules may constitute a large improvement in entailment accuracy, which can incite improvement in the SPRL model. 
-- Note: Shortly before the project deadline, a bug was found in the averaging algorithm, which impacted performance greatly: applicability values were assumed True/False instead of the actual yes/no values.
- As by far not all sentences in the Reisinger dataset had corresponding PropBank-roleset - VerbNet-class pairs in nltk, a large amount of data could not be labelled. Using newer or different resources may increase the available amount of data. 
- The PropBank predominantly contains financial data, and may therefore be suboptimally representative of the English language in general. The use of a different dataset, which matches the target distribution better, may improve performance and may be more robust towards out-of-distribution data.

## Acknowledgements
We thank our Professor, Dr. Anette Frank, as well as the Institute for Computational Linguistics of the Heidelberg University, for the support we were given and especially the access to resources like the PennTreeBank. This work would not have been possible if not for the written guidance of Reisinger et al. and White et al., who provided the SPR1 and SPR2 datasets. 
This project additonally powered by huggingface and its provided libraries, the PennTreeBank, PropBank, VerbNet and nltk.
### License
We give out our work under the MIT License.