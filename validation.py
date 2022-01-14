# -*- coding: utf-8 -*-
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm

"""## Config Define"""

# df_valid = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/train.csv")[
#     -64:
# ].reset_index(drop=True)

"""# Get Predictions"""

# Get start_logits and end_logits from model predict
# raw_predictions = (start_logits, end_logits)

"""# Compute Jaccard

## Postprocess
""" 

def getFeatureExampleIndex(features, examples):
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    return features_per_example


def getOffset(feature):
    return [
        (off if feature["sequence_ids"][i] == 1 else None)
        for i, off in enumerate(feature["offset_mapping"])
    ]


def validateAnswer(start_index, end_index, offset_mapping, max_answer_length):
    if (
        start_index >= len(offset_mapping)
        or end_index >= len(offset_mapping)
        or offset_mapping[start_index] is None
        or offset_mapping[end_index] is None
        or end_index < start_index
        or end_index - start_index + 1 > max_answer_length
    ):
        return False

    return True

def postprocess(
    examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30
):
    features_per_example = getFeatureExampleIndex(features, examples)

    predictions = collections.OrderedDict()

    all_start_logits, all_end_logits = raw_predictions

    for example_index, example examples.iterrows():
        feature_index = features_per_example[example_index]

        min_score = None
        valid_answers = []

        context = example["context"]
        for fi in feature_index:
            start_logits = all_start_logits[fi]
            end_logits = all_end_logits[fi]

            offset_mapping = getOffset(features[fi])

            cls_index = features[fi]["input_ids"].index(tokenizer.cls_token_id)
            feature_score = start_logits[cls_index] + end_logits[cls_index]
            if min_score is None or min_score < feature_score:
                min_score = feature_score

            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if validateAnswer(
                        start_index, end_index, offset_mapping, max_answer_length
                    ):

                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        valid_answers.append(
                            {
                                "score": start_logits[start_index]
                                + end_logits[end_index],
                                "text": context[start_char:end_char],
                            }
                        )

        best_answer = (
            sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            if valid_answers
            else {"text": "000", "score": 0.0}
        )

        predictions[example["id"]] = best_answer["text"]

    return predictions

"""## Compute"""

def jaccard(row):
    str1 = row[0]
    str2 = row[1]
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def computeJaccard(df_valid, start_logits, end_logits, tokenizer, max_seq_length=400, doc_stride=135):
    valid = df_valid.copy(deep=True)

    valid_features = []
    for i, row in valid.iterrows():
        valid_features += getFeatures(row, tokenizer, max_seq_length, doc_stride)
    
    final_predictions = postprocess(
        valid, valid_features, (start_logits, end_logits), tokenizer
    )

    res = valid.loc[:, ["id", "answer_text"]]
    res.rename(columns={"answer_text": "answer"}, inplace=True)
    res["prediction"] = res["id"].apply(lambda r: final_predictions[r])
    res["jaccard"] = res[["answer", "prediction"]].apply(jaccard, axis=1)

    return res.jaccard.mean(), res