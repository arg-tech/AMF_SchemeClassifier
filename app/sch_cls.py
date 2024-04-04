import torch
from datasets import Dataset
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification
from amf_fast_inference import model
from torch.nn import functional as F
import numpy as np
import json


TOKENIZER = AutoTokenizer.from_pretrained("raruidol/SchemeClassifier3-ENG")
MODEL = AutoModelForSequenceClassification.from_pretrained("raruidol/SchemeClassifier3-ENG")
#model_loader = model.ModelLoader("raruidol/SchemeClassifier3-ENG")
#MODEL = model_loader.load_model()


def preprocess_data(filexaif):
    inferences_id = []
    data = {'text': []}

    for node in filexaif['nodes']:
        if node['type'] == 'RA':
            # Uncomment for classifying the unspecified inferences only
            # if node['text'] == 'Default Inference':
            id = node['nodeID']
            inferences_id.append(id)
            for edge in filexaif['edges']:
                if edge['fromID'] == id:
                    claim_id = edge['toID']
                elif edge['toID'] == id:
                    premise_id = edge['fromID']

            for nd in filexaif['nodes']:
                if nd['nodeID'] == claim_id:
                    claim = nd['text']
                if nd['nodeID'] == premise_id:
                    premise = nd['text']

            data['text'].append(premise+'. '+claim)

    final_data = Dataset.from_dict(data)

    return final_data, inferences_id


def tokenize_sequence(samples):
    return TOKENIZER(samples["text"], padding=True, truncation=True)


def make_predictions(trainer, tknz_data):
    predicted_logprobs = trainer.predict(tknz_data)
    labels = []
    for sample in predicted_logprobs.predictions:
        torch_logits = torch.from_numpy(sample)
        probabilities = F.softmax(torch_logits, dim=-1).numpy()
        valid_check = probabilities > 0.95
        if True in valid_check:
            labels.append(np.argmax(sample, axis=-1))
        else:
            labels.append(-1)

    return labels


def output_xaif(idents, labels, fileaif):
    mapping_label = {-1: "Default Inference", 0: "Position to Know Argument", 1: "Ad Hominem Argument", 2: "Popular Acceptance",
       3: "Defeasible Rule-based Argument", 4: "Argument Based on Cases", 5: "Chained Argument from Rules and Cases",
       6: "Discovery Argument", 7: "Practical Reasoning"}
    for i in range(len(labels)):
        lb = mapping_label[labels[i]]
        id = idents[i]
        for node in fileaif['AIF']['nodes']:
            if node['nodeID'] == id:
                node['text'] = lb
                break
    return fileaif


def scheme_classification(xaif):

    # Generate a HF Dataset from all the "I" node pairs to make predictions from the xAIF file 
    # and a list of tuples with the corresponding "I" node ids to generate the final xaif file.
    dataset, ids = preprocess_data(xaif['AIF'])

    # Tokenize the Dataset.
    tokenized_data = dataset.map(tokenize_sequence, batched=True)

    # Instantiate HF Trainer for predicting.
    trainer = Trainer(MODEL)

    # Predict the list of labels for all the pairs of "I" nodes.
    labels = make_predictions(trainer, tokenized_data)

    # Prepare the xAIF output file.
    out_xaif = output_xaif(ids, labels, xaif)

    return out_xaif


# DEBUGGING:
if __name__ == "__main__":
    ff = open('../data.json', 'r')
    content = json.load(ff)
    # print(content)
    out = scheme_classification(content)
    # print(out)
    with open("../data_out.json", "w") as outfile:
        json.dump(out, outfile, indent=4)

