import json
import sys
import spacy
import os
import logging
import codecs
import random
import csv
import errno

base_path = 'results/sentiments'
faz_path = os.path.join(base_path, 'faz_sentiments.json')
sueddeutsche_path = os.path.join(base_path, 'sueddeutsche_sentiments.json')
zeit_path = os.path.join(base_path, 'zeit_sentiments.json')
spiegel_path = os.path.join(base_path, 'spiegel_sentiments.json')
welt_path = os.path.join(base_path, 'welt_sentiments.json')

save = 'results/sentiments/evaluation'

file_mapping = {'faz': faz_path,
           'sueddeutsche': sueddeutsche_path,
           'zeit': zeit_path,
           'spiegel': spiegel_path,
           'welt': welt_path
           }

logger = logging.getLogger(__name__)

def read_file_to_json(file_name):
    with open(file_name, 'r') as file:
        return json.load(file)

def analyze_score(sentiments, thresholds):
    sentiment_scores = {}

    extremely_negative = 0
    negative = 0
    neutral = 0
    positive = 0
    extremely_positive = 0

    for score in sentiments:
        if score < thresholds["extremely_negative"]:
            extremely_negative += 1
        elif score < thresholds["negative"]:
            negative += 1
        elif score < thresholds["neutral"]:
            neutral += 1
        elif score < thresholds["positive"]:
            positive += 1
        else:
            extremely_positive += 1

    relative = {}
    relative["extremely_negative"] = "{:.5f}".format(extremely_negative / len(sentiments))
    relative["negative"] = "{:.5f}".format(negative / len(sentiments))
    relative["neutral"] = "{:.5f}".format(neutral / len(sentiments))
    relative["positive"] = "{:.5f}".format(positive / len(sentiments))
    relative["extremely_positive"] = "{:.5f}".format(extremely_positive / len(sentiments))

    raw = {}
    raw["amount"] = len(sentiments)
    raw["extremely_negative"] = extremely_negative
    raw["negative"] = negative
    raw["neutral"] = neutral
    raw["positive"] = positive
    raw["extremely_positive"] = extremely_positive


    sentiment_scores["raw"] = raw
    sentiment_scores["relative"] = relative

    return sentiment_scores


def analyze_sentiments(site, sentiments, thresholds):
    scores = {}

    scores["comments"] = analyze_score(sentiments['comments'], thresholds)
    scores["sentences"] = analyze_score(sentiments['sentences'], thresholds)
    scores["words"] = analyze_score(sentiments['words'], thresholds)

    return scores

def evaluate_sentiments(sites, items=None):
    ##create histogram values based on scores

    for site in sites:
        site_scores = {}
        thresholds = {
            "extremely_negative": -0.8,
            "negative": -0.2,
            "neutral": 0.2,
            "positive": 0.8,
        }

        folder_name = "[{},{},{},{}]".format(thresholds["extremely_negative"], thresholds["negative"], thresholds["neutral"], thresholds["positive"])
        output_folder = os.path.join(save, folder_name)
        try:
            os.makedirs(output_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Evaluation already exists!")
                raise

        print("Starting Sentiment Evaluation for site: {}".format(site))
        sentiments = read_file_to_json(file_mapping[site])
        site_scores = analyze_sentiments(site, sentiments, thresholds)

        with codecs.open(os.path.join(output_folder, site + "_evaluation.json"), 'w', encoding='utf-8') as file:
            json.dump(site_scores, file, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)

if __name__ == "__main__":
    evaluate_sentiments(['faz', 'zeit', 'spiegel', 'welt', 'sueddeutsche'])
    #evaluate_sentiments(['faz'])
