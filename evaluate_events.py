import datetime
import json
import os

import pandas as pd
from sklearn.metrics import *
from collections import defaultdict

from TwitterEventDetector import TwitterEventDetector


def cluster_event_match(labels, preds):
    data = pd.DataFrame()
    data["pred"] = preds
    data["label"] = labels

    match = data.groupby(["label", "pred"], sort=False).size().reset_index(name="a")
    b, c = [], []
    for idx, row in match.iterrows():
        b_ = ((data["label"] != row["label"]) & (data["pred"] == row["pred"]))
        b.append(b_.sum())
        c_ = ((data["label"] == row["label"]) & (data["pred"] != row["pred"]))
        c.append(c_.sum())
    match["b"] = pd.Series(b)
    match["c"] = pd.Series(c)
    # recall = nb true positive / (nb true positive + nb false negative)
    match["r"] = match["a"] / (match["a"] + match["c"])
    # precision = nb true positive / (nb true positive + nb false positive)
    match["p"] = match["a"] / (match["a"] + match["b"])
    match["f1"] = 2 * match["r"] * match["p"] / (match["r"] + match["p"])
    match = match.sort_values("f1", ascending=False)
    # macro_average_f1 = match.drop_duplicates("label").f1.mean()
    # macro_average_precision = match.drop_duplicates("label").p.mean()
    # macro_average_recall = match.drop_duplicates("label").r.mean()

    match = match.drop_duplicates("label")
    sizes = match[['a', 'b', 'c']].sum(axis=1)
    proportions = sizes / sizes.sum()
    micro_average_f1 = (match.f1 * proportions).sum()
    micro_average_precision = (match.p * proportions).sum()
    micro_average_recall = (match.r * proportions).sum()
    return micro_average_precision, micro_average_recall, micro_average_f1, match


def print_statistics(tweet_ids, new_preds2, tweets_with_label):
    results = pd.DataFrame()
    results['tweet_id'] = tweet_ids
    results['pred'] = new_preds2
    tweet_with_label_set = dict(zip(tweets_with_label['tweet_id'].astype(str), tweets_with_label['label']))

    rows = []
    tweets_with_labels = 0
    label_to_tweets = defaultdict(list)
    for t, c in zip(results['tweet_id'], results['pred']):
        if t in tweet_with_label_set:
            rows.append([t, tweet_with_label_set[t], c])
            tweets_with_labels += 1
            label_to_tweets[tweet_with_label_set[t]].append(t)
        # else:
        #     label_to_tweets[-1].append(t)
        #     rows.append([t, -1, c])

    output = pd.DataFrame(rows, columns=['tweet_id', 'label', 'pred'])

    ami = adjusted_mutual_info_score(output['label'], output['pred'])
    ari = adjusted_rand_score(output['label'], output['pred'])
    nmi = normalized_mutual_info_score(output['label'], output['pred'])

    print('adjusted_mutual_info', ami)
    print('adjusted_rand', ari)
    print('Normalized Mutal Info:', nmi)
    macro_p, macro_r, macro_f1, match = cluster_event_match(output['label'], output['pred'])
    print('micro_p:', macro_p)
    print('micro_r:', macro_r)
    print('micro_f1:', macro_f1)
    print('#labeled tweets:', tweets_with_labels, )
    print('#labels:', len(set(output['label'])), '#preds:', len(set(output['pred'])))


def read_tweets_from_file(file_path):
    input_tweets = []
    with open(file_path) as f:
        for line in f:
            input_tweets.append(json.loads(line))
    return input_tweets


def create_labels_file(start_date_str, end_date_str):
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
    day_count = (end_date - start_date).days
    current_date = start_date

    tweets_with_label = pd.read_csv('data/cleaned_tweets/event_2012_relevant_tweets.tsv',
                                    sep='\t', header=None, names=['label', 'tweet_id'])
    tweets_with_label = tweets_with_label.drop_duplicates()

    relevant_tweet_ids = set()
    tweet_with_labels = set(tweets_with_label['tweet_id'].astype(str))
    for d in range(day_count):
        date_str = current_date.strftime("%Y-%m-%d")
        print(date_str)
        subwindow_dir = f'data/cleaned_tweets/without_retweets/{date_str}/'
        subwindow_files = [f.name for f in os.scandir(subwindow_dir) if f.is_file()]
        for subwindow_name in subwindow_files:
            tweet_jsons = read_tweets_from_file(subwindow_dir + subwindow_name)
            for tweet_json in tweet_jsons:
                if tweet_json['tweet_id'] in tweet_with_labels:
                    relevant_tweet_ids.add(tweet_json['tweet_id'])
            pass
        current_date = current_date + datetime.timedelta(days=1)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    tweets_with_label[tweets_with_label['tweet_id'].astype(str).isin(relevant_tweet_ids)].to_csv(
        f'event_2012_relevant_tweets_{start_str}_{end_str}.tsv',
        sep='\t', header=None, index=False)


def load_event_files(start_date_str='2012-10-11', end_date_str='2012-10-22', folder_name='results'):
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
    day_count = (end_date - start_date).days
    current_date = start_date

    # tweets_with_label = pd.read_csv('data/cleaned_tweets/event_2012_relevant_tweets.tsv',
    #                                 sep='\t', header=None, names=['label', 'tweet_id'])
    # tweets_with_label = tweets_with_label.drop_duplicates()
    dfs = []
    for d in range(day_count):
        date_str = current_date.strftime("%Y-%m-%d")
        results_df = pd.read_csv(f'{folder_name}/{date_str}/events.csv').drop_duplicates(['tweet_id'])
        current_date = current_date + datetime.timedelta(days=1)
        dfs.append(results_df)
    return pd.concat(dfs, axis=0)


start_date_str = '2012-10-11'
end_date_str = '2012-10-22'
# create_labels_file(start_date_str, end_date_str)
tweets_with_label = pd.read_csv(f'event_2012_relevant_tweets_{start_date_str}_{end_date_str}.tsv',
                                sep='\t', header=None, names=['label', 'tweet_id'])
# tweets_with_label = pd.read_csv('data/cleaned_tweets/event_2012_relevant_tweets.tsv',
#                                     sep='\t', header=None, names=['label', 'tweet_id'])
# tweets_with_label = tweets_with_label.drop_duplicates()
# results_df = pd.read_csv('results/2012-10-10/events.csv')
results_df = load_event_files(start_date_str=start_date_str, end_date_str=end_date_str, folder_name='ensemble_results')
tweet_ids = results_df['tweet_id'].astype(str)
new_preds2 = results_df['label']
print_statistics(tweet_ids, new_preds2, tweets_with_label)
