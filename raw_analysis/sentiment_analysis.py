import json
import sys
import spacy
import os
import logging
import codecs
import random
import csv

base_path = '../germannewssites/datasets/data'
faz_path = os.path.join(base_path, 'faz_nlp.json')
sueddeutsche_path = os.path.join(base_path, 'sueddeutsche_nlp.json')
zeit_path = os.path.join(base_path, 'zeit_nlp.json')
spiegel_path = os.path.join(base_path, 'spiegel_nlp.json')
welt_path = os.path.join(base_path, 'welt_nlp.json')

save = 'results/sentiments'

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

def analyze_words(comments, sentiment_map, pos_list):

    sentiment_scores = []
    for doc in comments:
        del doc["target"]

        for sentence in doc:
            for token in doc[sentence]["token"]:
                if token["pos"] in pos_list:
                    try:
                        sentiment_scores.append(float(sentiment_map[token["lemma"]]))
                    except KeyError:
                        # Key is not present
                        continue

    return sentiment_scores

def analyze_sentences(comments, sentiment_map, pos_list):

    sentiment_scores = []
    for doc in comments:
        for sentence in doc:
            sentiment_score = 0.0
            word_counter = 0
            word_miss = 0
            for token in doc[sentence]["token"]:
                if token["pos"] in pos_list:
                    try:
                        sentiment_score += float(sentiment_map[token["lemma"]])
                        word_counter += 1
                    except KeyError:
                        continue

            if sentiment_score != 0.0 and word_counter != 0:
                sentence_average_sentiment = sentiment_score / word_counter
                sentiment_scores.append(sentence_average_sentiment)
            else:
                sentiment_scores.append(sentiment_score)
    return sentiment_scores

def analyze_comments(comments, sentiment_map, pos_list):

    sentiment_scores = []
    for doc in comments:
        sentiment_score = 0.0
        word_counter = 0

        for sentence in doc:
            for token in doc[sentence]["token"]:
                if token["pos"] in pos_list:
                    try:
                        sentiment_score += float(sentiment_map[token["lemma"]])
                        word_counter += 1
                    except KeyError:
                        continue

        if sentiment_score != 0.0 and word_counter != 0:
            comment_average_sentiment = sentiment_score / word_counter
            sentiment_scores.append(comment_average_sentiment)
        else:
            sentiment_scores.append(sentiment_score)

    return sentiment_scores

def analyze_hit_rate(comments, sentiment_map, pos_list):

    counter = 0
    misses = 0
    for doc in comments:
        del doc["target"]

        for sentence in doc:
            for token in doc[sentence]["token"]:
                if token["pos"] in pos_list:
                    try:
                        counter += 1
                        sentiment_score = float(sentiment_map[token["lemma"]])
                    except KeyError:
                        misses += 1

    return 1.0 - (misses / counter)

def analyze_site(site, comments, sentiment_map):
    scores = {}
    pos_list = ['VERB', 'NOUN', 'ADJ', 'ADV']
    print ("Comments in File: {}".format(len(comments)))

    scores["comments"] = len(comments)
    scores["words"] = analyze_words(comments, sentiment_map, pos_list)
    scores["sentences"] = analyze_sentences(comments, sentiment_map, pos_list)
    scores["comments"] = analyze_comments(comments, sentiment_map, pos_list)

    return scores

def get_hit_rate(site, comments, sentiment_map):
    scores = {}
    pos_list = ['VERB', 'NOUN', 'ADJ', 'ADV']
    print ("Comments in File: {}".format(len(comments)))

    return analyze_hit_rate(comments, sentiment_map, pos_list)

def evaluate_hit_rate(sites, items=None):
    sentiment_map = {}

    with open('SentiWS_v1.8c_Positive.txt', 'r') as f:
        senti_pos = csv.reader(f, delimiter='\t')
        for senti_score in senti_pos:
            sentiment_map[senti_score[0].split('|')[0].lower()] = senti_score[1]

            try:
                derivations = senti_score[2].split(',')
                for derivation in derivations:
                    sentiment_map[derivation.lower()] = senti_score[1]
            except IndexError:
                # Key is not present
                pass

    with open('SentiWS_v1.8c_Negative.txt', 'r') as f:
        senti_neg = csv.reader(f, delimiter='\t')
        for senti_score in senti_neg:
            sentiment_map[senti_score[0].split('|')[0].lower()] = senti_score[1]

            try:
                derivations = senti_score[2].split(',')
                for derivation in derivations:
                    sentiment_map[derivation.lower()] = senti_score[1]
            except IndexError:
                # Key is not present
                pass

    site_scores = {}

    #with codecs.open(os.path.join(save, "sentiment_map.txt"), 'w', encoding='utf-8') as file:
    #    json.dump(sentiment_map, file, ensure_ascii=False)

    for site in sites:
        comment_items = read_file_to_json(file_mapping[site])
        comments = [v for k,v in comment_items.items()]

        print ("Hit rate sentiWS for {} = {}".format(site, get_hit_rate(site, comments, sentiment_map)))

def senti_ws(sites, items=None):
    sentiment_map = {}

    with open('SentiWS_v1.8c_Positive.txt', 'r') as f:
        senti_pos = csv.reader(f, delimiter='\t')
        for senti_score in senti_pos:
            sentiment_map[senti_score[0].split('|')[0].lower()] = senti_score[1]

            try:
                derivations = senti_score[2].split(',')
                for derivation in derivations:
                    sentiment_map[derivation.lower()] = senti_score[1]
            except IndexError:
                # Key is not present
                pass

    with open('SentiWS_v1.8c_Negative.txt', 'r') as f:
        senti_neg = csv.reader(f, delimiter='\t')
        for senti_score in senti_neg:
            sentiment_map[senti_score[0].split('|')[0].lower()] = senti_score[1]

            try:
                derivations = senti_score[2].split(',')
                for derivation in derivations:
                    sentiment_map[derivation.lower()] = senti_score[1]
            except IndexError:
                # Key is not present
                pass

    site_scores = {}

    #with codecs.open(os.path.join(save, "sentiment_map.txt"), 'w', encoding='utf-8') as file:
    #    json.dump(sentiment_map, file, ensure_ascii=False)

    for site in sites:
        print("Starting Analysis for Sentiment Score for site: {}".format(site))
        comment_items = read_file_to_json(file_mapping[site])
        comments = [v for k,v in comment_items.items()]

        site_scores = analyze_site(site, comments, sentiment_map)

        with codecs.open(os.path.join(save, site + "_sentiments.json"), 'w', encoding='utf-8') as file:
            json.dump(site_scores, file, ensure_ascii=False)

''' this script creates a json which contains all sentiments for each comment, sentence, and word of a site '''
if __name__ == "__main__":
    #senti_ws(['faz', 'zeit', 'spiegel', 'welt', 'sueddeutsche'])
    evaluate_hit_rate(['faz', 'zeit', 'spiegel', 'welt', 'sueddeutsche'])
