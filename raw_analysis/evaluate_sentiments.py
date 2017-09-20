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
def analyze_score_intervals(sentiments, thresholds):
    zero = 0
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0
    six = 0
    seven = 0
    eight = 0
    nine = 0
    ten = 0
    eleven = 0
    twelve = 0
    thirteen = 0
    fourteen = 0
    fivteen = 0
    sixteen = 0
    seventeen = 0
    eighteen = 0
    nineteen = 0

    for score in sentiments:
        if score < thresholds["0"]:
            zero += 1
        elif score < thresholds["1"]:
            one += 1
        elif score < thresholds["2"]:
            two += 1
        elif score < thresholds["3"]:
            three += 1
        elif score < thresholds["4"]:
            four += 1
        elif score < thresholds["5"]:
            five += 1
        elif score < thresholds["6"]:
            six += 1
        elif score < thresholds["7"]:
            seven += 1
        elif score < thresholds["8"]:
            eight += 1
        elif score < thresholds["9"]:
            nine += 1
        elif score < thresholds["10"]:
            ten += 1
        elif score < thresholds["11"]:
            eleven += 1
        elif score < thresholds["12"]:
            twelve += 1
        elif score < thresholds["13"]:
            thirteen += 1
        elif score < thresholds["14"]:
            fourteen += 1
        elif score < thresholds["15"]:
            fivteen += 1
        elif score < thresholds["16"]:
            sixteen += 1
        elif score < thresholds["17"]:
            seventeen += 1
        elif score < thresholds["18"]:
            eighteen += 1
        else:
            nineteen += 1

    relative = {}
    relative["amount"] = len(sentiments)
    relative["0"] = "{:.5f}".format(zero / len(sentiments))
    relative["1"] = "{:.5f}".format(one / len(sentiments))
    relative["2"] = "{:.5f}".format(two / len(sentiments))
    relative["3"] = "{:.5f}".format(three / len(sentiments))
    relative["4"] = "{:.5f}".format(four / len(sentiments))
    relative["5"] = "{:.5f}".format(five / len(sentiments))
    relative["6"] = "{:.5f}".format(six / len(sentiments))
    relative["7"] = "{:.5f}".format(seven / len(sentiments))
    relative["8"] = "{:.5f}".format(eight / len(sentiments))
    relative["9"] = "{:.5f}".format(nine / len(sentiments))
    relative["10"] = "{:.5f}".format(ten / len(sentiments))
    relative["11"] = "{:.5f}".format(eleven / len(sentiments))
    relative["12"] = "{:.5f}".format(twelve / len(sentiments))
    relative["13"] = "{:.5f}".format(thirteen / len(sentiments))
    relative["14"] = "{:.5f}".format(fourteen / len(sentiments))
    relative["15"] = "{:.5f}".format(fivteen / len(sentiments))
    relative["16"] = "{:.5f}".format(sixteen / len(sentiments))
    relative["17"] = "{:.5f}".format(seventeen / len(sentiments))
    relative["18"] = "{:.5f}".format(eighteen / len(sentiments))
    relative["19"] = "{:.5f}".format(nineteen / len(sentiments))

    return relative

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


def analyze_intervals(site, sentiments, thresholds):
    scores = {}

    scores["comments"] = analyze_score_intervals(sentiments['comments'], thresholds)
    scores["sentences"] = analyze_score_intervals(sentiments['sentences'], thresholds)
    scores["words"] = analyze_score_intervals(sentiments['words'], thresholds)

    return scores

def evaluate_sentiments(sites, items=None):
    ##create histogram values based on scores

    for site in sites:
        site_scores = {}
        thresholds = {
            "extremely_negative": -0.6,
            "negative": -0.2,
            "neutral": 0.2,
            "positive": 0.6,
        }
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

def evaluate_average(sites, items=None):
    for site in sites:
        sentiments = read_file_to_json(file_mapping[site])

        sentiment_sum = 0.0
        for sentiment in sentiments['comments']:
            sentiment_sum += sentiment

        print ("Sentiment average for site {} is {}".format(site, (sentiment_sum / len(sentiments['comments']))))

def evaluate_intervals(sites, items=None):
    ##create histogram values based on scores

    for site in sites:
        site_scores = {}
        thresholds = {
            "0": -0.9,
            "1": -0.8,
            "2": -0.7,
            "3": -0.6,
            "4": -0.5,
            "5": -0.4,
            "6": -0.3,
            "7": -0.2,
            "8": -0.1,
            "9": 0.0,
            "10": 0.1,
            "11": 0.2,
            "12": 0.3,
            "13": 0.4,
            "14": 0.5,
            "15": 0.6,
            "16": 0.7,
            "17": 0.8,
            "18": 0.9,
        }

        folder_name = "0.1_intervals"
        output_folder = os.path.join(save, folder_name)
        try:
            os.makedirs(output_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Evaluation already exists!")
                raise

        print("Starting Sentiment Evaluation for site: {}".format(site))
        sentiments = read_file_to_json(file_mapping[site])
        site_scores = analyze_intervals(site, sentiments, thresholds)

        with codecs.open(os.path.join(output_folder, site + "_evaluation.json"), 'w', encoding='utf-8') as file:
            json.dump(site_scores, file, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)

''' evaluates the json files created by sentiment_analysis.py '''
if __name__ == "__main__":
    #evaluate_sentiments(['faz', 'zeit', 'spiegel', 'welt', 'sueddeutsche'])
    #evaluate_intervals(['faz', 'zeit', 'spiegel', 'welt', 'sueddeutsche'])
    evaluate_average(['faz', 'zeit', 'spiegel', 'welt', 'sueddeutsche'])
