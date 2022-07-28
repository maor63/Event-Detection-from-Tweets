import os
import timeit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
from collections import defaultdict, OrderedDict, deque, Counter
from pathlib import Path
import json
from tqdm.auto import tqdm
import datetime
import pandas as pd
import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
import re
import random
from sklearn.metrics import *
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.cluster import KMeans
from unidecode import unidecode
from datasketch import MinHash, MinHashLSH
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def run_lsh(texts, threshold=0.7, num_perm=128):
    word_sets = []
    for words in tqdm(texts.str.split()):
        if len(words) > 0:
            word_sets.append(pd.Series(words).str.encode('utf-8').tolist())
        else:
            word_sets.append(['empty'.encode('utf-8')])

    mhs = MinHash.bulk(word_sets, num_perm=num_perm)

    preds = np.zeros(len(texts))
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, mh in tqdm(enumerate(mhs, 0), total=len(mhs)):
        lsh.insert(i, mh)
        results = lsh.query(mh)
        new_l = results[0]
        for j in results:
            preds[j] = new_l
    return preds


def tf_idf_vecs(texts, max_features=5000):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X


def read_tweets_from_file(file_path):
    input_tweets = []
    with open(file_path) as f:
        for line in f:
            input_tweets.append(json.loads(line))
    return input_tweets

def main():
    start_date = datetime.datetime.strptime('2012-10-11', "%Y-%m-%d")
    end_date = datetime.datetime.strptime('2012-10-12', "%Y-%m-%d")
    day_count = (end_date - start_date).days
    current_date = start_date
    output_dir = 'ensemble_kmeans_2000_DBSCAN_0-4'
    event_no = 0
    label_no = 0
    subwindow_size = 2
    n_clusters = 2000
    all_preds = []
    all_tweets = []
    for d in range(day_count):
        start = timeit.default_timer()
        date_str = current_date.strftime("%Y-%m-%d")
        print('Current date:', date_str)
        subwindow_dir = f'data/cleaned_tweets/without_retweets/{date_str}/'
        event_output_dir = f'{output_dir}/{date_str}/'
        subwindow_files = [f.name for f in os.scandir(subwindow_dir) if f.is_file()]

        for idx, subwindow_names in enumerate(chunker(subwindow_files, subwindow_size)):
            print(idx)
            tweets = []
            for subwindow_name in subwindow_names:
                tweets += read_tweets_from_file(subwindow_dir + subwindow_name)
            tweets_df = pd.DataFrame(tweets)
            texts = tweets_df.text
            tweet_ids = tweets_df.tweet_id.tolist()
            old_preds = run_lsh(texts, threshold=0.5, num_perm=64)
            # clustering = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, verbose=0).fit(X)
            clustering_method = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, verbose=0)
            new_preds1 = cluster_tweets(clustering_method, old_preds, texts)

            clustering_method = DBSCAN(eps=0.6, min_samples=1, metric='cosine')
            new_preds2 = cluster_tweets(clustering_method, new_preds1, texts)

            new_preds2 = list(np.array(new_preds2) + event_no)
            all_preds += new_preds2
            event_no = len(set(all_preds))
            all_tweets += tweet_ids

            results_df = pd.DataFrame()
            results_df['tweet_id'] = tweet_ids
            results_df['label'] = new_preds2

            if os.path.exists(event_output_dir + 'events.csv'):
                results_df.to_csv(event_output_dir + 'events.csv', index=False, header=None, mode='a', encoding='utf-8')
            else:
                if not os.path.exists(event_output_dir):
                    os.makedirs(event_output_dir)
                results_df.to_csv(event_output_dir + 'events.csv', index=False)

        stop = timeit.default_timer()
        print('Time in minutes: ', (stop - start) / 60)
        current_date = current_date + datetime.timedelta(days=1)


def cluster_tweets(clustering_method, old_preds, texts):
    cluster_representative = get_cluster_representative_text(old_preds, texts)
    # X = sentence_transformer.encode(list(representative_m.values()))
    X = tf_idf_vecs(cluster_representative.values(), max_features=5000)
    cluster_start = timeit.default_timer()
    clustering = clustering_method.fit(X)
    sub_preds = clustering.labels_
    cluster_end = timeit.default_timer()
    print('cluster time sec:', cluster_end - cluster_start)
    new_preds = update_clusters(sub_preds, old_preds, cluster_representative)
    return new_preds


def update_clusters(new_preds, old_preds, cluster_representative):
    old_to_new_labels = {}
    base = len(cluster_representative.keys())
    for old_l, new_l in zip(cluster_representative.keys(), new_preds):
        if new_l != -1:
            old_to_new_labels[old_l] = base + new_l
    return [old_to_new_labels.get(p, p) for p in old_preds]


def get_cluster_representative_text(preds, texts):
    m = defaultdict(list)
    for i, p in enumerate(preds):
        m[p].append(i)
    representative_m = {}
    for l, vals in m.items():
        representative_m[l] = texts[max(vals, key=lambda x: len(texts[x].split()))]
    return representative_m


if __name__ == '__main__':
    main()
