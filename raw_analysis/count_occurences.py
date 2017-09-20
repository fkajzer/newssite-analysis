import json
import sys
import ijson
from collections import Counter
import os, codecs
import math
import spacy
import csv

base_path = '../germannewssites/raw_data'
faz_path = os.path.join(base_path, 'faz/faz.json')
sueddeutsche_path = os.path.join(base_path, 'sueddeutsche/sueddeutsche.json')
zeit_path = os.path.join(base_path, 'zeit/zeit.json')
spiegel_path = os.path.join(base_path, 'spiegel/spiegel.json')
welt_path = os.path.join(base_path, 'welt/welt.json')

file_mapping = {'faz': faz_path,
           'sueddeutsche': sueddeutsche_path,
           'zeit': zeit_path,
           'spiegel': spiegel_path,
           'welt': welt_path
           }

save = 'results/sites'

def read_file_to_json(file_name):
    with open(file_name, 'r') as file:
        return json.load(file)

def analyze_sentiment(nlp, sentiment_map, comment):
    processed = nlp(comment)
    senti_score = 0.0

    for token in processed:
        if token.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV']:
            try:
                senti_score += float(sentiment_map[token.lemma_])
            except KeyError:
                # Key is not present
                continue
    return senti_score

def process_welt(comments):
    only_comments = []
    only_profiles = []

    for comment in comments:
        try:
            a_comment = comment["contents"]
            only_comments.append(comment)
        except:
            pass

    site_scores = {}
    site_scores["site_analysis"] = analyze_site("welt", only_comments)

    return site_scores

def get_top_users_for_site(site, comments, top_n):
    if site == 'sueddeutsche':
        return Counter(comment['author']['name'] for comment in comments).most_common(top_n)
    if site == "welt":
        return Counter(comment['user']['displayName'] for comment in comments).most_common(top_n)

    return Counter(comment['user_name'] for comment in comments).most_common(top_n)

def analyze_all_comments(site, all_comments_len, comments):
    user_comment_scores = {}

    user_comment_length = 0
    user_comment_upvotes = 0
    user_comment_replies = 0

    for comment in comments:
        if site == "faz":
            for quote in comment["quote"]:
                user_comment_replies += 1
            user_comment_upvotes += int(comment["upvotes"])
            user_comment_length += len(comment["content"].split(" "))
        if site == "zeit":
            if comment["quote"] is not None:
                user_comment_replies += 1
            user_comment_upvotes += int(comment["upvotes"])
            user_comment_length += len(comment["content"].split(" "))
        if site == "spiegel":
            if comment["quote"] is not None:
                user_comment_replies += 1
            user_comment_length += len(comment["content"].split(" "))
        if site == "welt":
            user_comment_upvotes += int(comment["likes"])
            user_comment_length += len(comment["contents"].split(" "))
            try:
                parent = comment["parentId"]
                user_comment_replies += 1
            except:
                pass
        if site == "sueddeutsche":
            if comment["parent"] is not None:
                user_comment_replies += 1
            user_comment_upvotes += int(comment["likes"])
            user_comment_length += len(comment["raw_message"].split(" "))

    user_comment_scores["user_comment_length"] = user_comment_length
    user_comment_scores["user_comment_replies"] = user_comment_replies
    if site != "spiegel":
        user_comment_scores["user_comment_upvotes"] = user_comment_upvotes

    user_comment_scores["average_comment_length"] = "{:.2f}".format(user_comment_length / all_comments_len)
    user_comment_scores["average_comment_replies"] = "{:.2f}".format(user_comment_replies / all_comments_len)
    if site != "spiegel":
        user_comment_scores["average_comment_upvotes"] = "{:.2f}".format(user_comment_upvotes / all_comments_len)

    return user_comment_scores

def analyze_top_user(site, all_comment_results, comments):
    nlp = spacy.load('de')
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

    top_user_scores = {}
    top_users = get_top_users_for_site(site, comments, 20)

    top_users_scores = {}

    for user, comment_len in top_users:
        user_comment_scores = {}
        user_comment_scores["comments"] = comment_len

        user_comment_length = 0
        user_comment_upvotes = 0
        user_comment_replies = 0
        user_comment_sentiment = 0.0

        for comment in comments:
            if (site not in ["welt", "sueddeutsche", "faz"] and comment["user_name"] == user) or (site == "welt" and comment['user']['displayName'] == user) or (site == "sueddeutsche" and comment['author']['name'] == user):
                if site == "zeit":
                    user_comment_sentiment += analyze_sentiment(nlp, sentiment_map, comment["content"])
                    if comment["quote"] is not None:
                        user_comment_replies += 1
                    user_comment_upvotes += int(comment["upvotes"])
                    user_comment_length += len(comment["content"].split(" "))
                if site == "spiegel":
                    user_comment_sentiment += analyze_sentiment(nlp, sentiment_map, comment["content"])
                    if comment["quote"] is not None:
                        user_comment_replies += 1
                    user_comment_length += len(comment["content"].split(" "))
                if site == "welt":
                    user_comment_sentiment += analyze_sentiment(nlp, sentiment_map, comment["contents"])
                    user_comment_upvotes += int(comment["likes"])
                    user_comment_length += len(comment["contents"].split(" "))
                    try:
                        parent = comment["parentId"]
                        user_comment_replies += 1
                    except:
                        pass
                if site == "sueddeutsche":
                    if comment["parent"] is not None:
                        user_comment_replies += 1
                    user_comment_sentiment += analyze_sentiment(nlp, sentiment_map, comment["raw_message"])
                    user_comment_upvotes += int(comment["likes"])
                    user_comment_length += len(comment["raw_message"].split(" "))
            elif site == "faz":
                if user == comment['user_name']:
                    user_comment_sentiment += analyze_sentiment(nlp, sentiment_map, comment["content"])
                    user_comment_upvotes += int(comment["upvotes"])
                    user_comment_length += len(comment["content"].split(" "))
                for quote in comment["quote"]:
                    if quote['user_name'] == user:
                        user_comment_replies += 1

        if site != "spiegel":
            user_comment_scores["average_comment_upvotes"] = "{:.2f}".format(user_comment_upvotes / comment_len)
        user_comment_scores["average_comment_sentiment"] = "{:.2f}".format(user_comment_sentiment / comment_len)
        user_comment_scores["average_comment_length"] = "{:.2f}".format(user_comment_length / comment_len)
        user_comment_scores["average_comment_replies"] = "{:.2f}".format(user_comment_replies / comment_len)

        user_comment_scores["comment_share"] = "{:.5f}".format(comment_len / len(comments))
        if site != "spiegel":
            user_comment_scores["upvote_share"] = "{:.5f}".format(user_comment_upvotes / all_comment_results["user_comment_upvotes"])
        user_comment_scores["word_share"] = "{:.5f}".format(user_comment_length / all_comment_results["user_comment_length"])
        user_comment_scores["reply_share"] = "{:.5f}".format(user_comment_replies / all_comment_results["user_comment_replies"])

        top_users_scores[user] = user_comment_scores
    return top_users_scores

def analyze_site(site, comments):
    print ("Comments in File: {}".format(len(comments)))
    scores = {}

    scores["comments"] = len(comments)

    if site == 'sueddeutsche':
        user_count = len(Counter(comment['author']['name'] for comment in comments).keys())
    elif site == "welt":
        user_count = len(Counter(comment['user']['displayName'] for comment in comments).keys())
    else:
        user_count = len(Counter(comment['user_name'] for comment in comments).keys())
    scores["users"] = user_count

    scores["comment_scores"] = analyze_all_comments(site, scores["comments"], comments)
    scores["top_user"] = analyze_top_user(site, scores["comment_scores"], comments)

    return scores

def counting_analysis(sites):
    for site in sites:
        print("Starting Analysis for site: {}".format(site))

        comment_items = read_file_to_json(file_mapping[site])
        if site == "welt":
            site_scores = process_welt(comment_items)
        else:
            site_scores = analyze_site(site, comment_items)

        with codecs.open(os.path.join(save, site + "_analysis.json"), 'w', encoding='utf-8') as file:
            json.dump(site_scores, file, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)

''' evaluates wordlength, reply_share, upvotes and comment_share for sites and their top users '''
if __name__ == "__main__":
    counting_analysis(['faz', 'spiegel', 'zeit', 'welt', 'sueddeutsche'])
    #counting_analysis(['welt'])
