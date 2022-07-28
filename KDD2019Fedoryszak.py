import datetime
import json
import os
import timeit
from collections import defaultdict
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import community.community_louvain
import networkx as nx
from collections import defaultdict, OrderedDict
from pathlib import Path
import json
import datetime
import re
import nltk
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pickle

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


class EntityExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def add_named_entity(self, tweet_json):
        text = re.sub(r'http\S+', '', tweet_json['text'])
        tweet_json['named_entities'] = list(map(str, (self.nlp(text).ents)))
        return tweet_json

    def get_entities_from_chunk(self, namedEnt):
        entities = []
        for chunk in namedEnt:
            if hasattr(chunk, 'label'):
                word = ' '.join(c for c, tag in chunk)
                entities.append(word)
        return entities

    def extract_hashtags(tweet_json):
        tags = tweet_json['text']
        return [tag.strip("#") for tag in tags.split() if tag.startswith("#")]

    def apply(self, tweet_jsons):
        named_entities = [self.add_named_entity(tweet) for tweet in tqdm(tweet_jsons, desc='extract entities')]
        return named_entities


class EntitySimilarityComputer:
    def __init__(self):
        pass

    def apply(self, tweet_jsons):
        vocab = set()
        corpus = []
        for tweet in tweet_jsons:
            vocab.update(tweet['named_entities'])
            vocab.update(tweet['entities']['hashtags'])
            corpus.append(tweet['text'])
        # vectorizer = CountVectorizer(vocabulary=vocab, ngram_range=(1, 2), analyzer='word', lowercase=False)
        vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', lowercase=True)
        tweet_vecs = vectorizer.fit_transform(corpus)
        X = np.transpose(tweet_vecs)
        similarities = cosine_similarity(X)
        return similarities, vectorizer.get_feature_names(), tweet_vecs


class EntityCluster:
    def __init__(self, threshold=0.0, resolution=1.0):
        self.threshold = threshold
        self.resolution = resolution

    def cluster(self, similarities):
        A = np.zeros_like(similarities)
        A[similarities > self.threshold] = 1
        graph = nx.from_numpy_matrix(A)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        partition = community.community_louvain.best_partition(graph, partition=None, weight='weight',
                                                               resolution=self.resolution, randomize=None,
                                                               random_state=123)
        return partition


def partition_to_cluster(partition, start_from=0):
    clusters = defaultdict(list)
    for node, label in partition.items():
        clusters[label + start_from].append(node)

    return OrderedDict(sorted(clusters.items()))


class ClusterLinker:
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        pass

    def sum_count_by_cluster(self, X, clustering):
        return np.stack([X[tweets_idx, :].sum(axis=0) for c, tweets_idx in clustering.items()])

    def link(self, clustering1, clustering2, tweet_jsons1, tweet_jsons2):
        vocab = set()
        corpus1, corpus2 = [], []

        for tweet1 in tweet_jsons1:
            vocab.update(tweet1['named_entities'])
            corpus1.append(tweet1['text'])

        for tweet2 in tweet_jsons2:
            vocab.update(tweet2['named_entities'])
            corpus2.append(tweet2['text'])

        vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', lowercase=True)
        X = vectorizer.transform(corpus1)
        Y = vectorizer.transform(corpus2)

        X_agg = self.sum_count_by_cluster(X, clustering1)
        Y_agg = self.sum_count_by_cluster(Y, clustering2)

        similarities = cosine_similarity(X_agg, Y_agg)
        # (n_samples_X, n_samples_Y)
        # print(similarities, vectorizer.get_feature_names())
        # print(list(zip(*np.where(similarities > self.threshold))))

        edges = [(v1, v2 + X_agg.shape[0]) for v1, v2 in zip(*np.where(similarities > self.threshold))]
        # print(edges)
        G = nx.Graph()
        G.add_nodes_from(
            [(v, {'bipartite': 0 if v < X_agg.shape[0] else 1}) for v in range(X_agg.shape[0] + Y_agg.shape[0])])
        G.add_edges_from(edges)
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        # print(nx.is_connected(G))
        # nx.draw(G)
        # plt.show()
        u = [n for n in G.nodes if G.nodes[n]['bipartite'] == 0]
        matching = nx.bipartite.maximum_matching(G, top_nodes=u)
        return {c1: c2 - X_agg.shape[0] for c1, c2 in matching.items() if c1 < c2}


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def read_tweets_from_file(file_path):
    input_tweets = []
    with open(file_path) as f:
        for line in f:
            input_tweets.append(json.loads(line))
    return input_tweets


def get_clustering_vectors(clustering, size):
    vecs = []
    for c, entities_idx in clustering.items():
        vec = np.zeros(size)
        vec[entities_idx] = 1
        vecs.append(vec)
    return np.stack(vecs)


def main():
    start_date = datetime.datetime.strptime('2012-10-11', "%Y-%m-%d")
    end_date = datetime.datetime.strptime('2012-10-12', "%Y-%m-%d")
    day_count = (end_date - start_date).days
    current_date = start_date
    subwindow_size = 2

    S = 0.1
    R = 1.0
    output_dir = 'KDD2019Fedoryszak/entity_thresh_0-1'
    total_tweet_clusters = []

    entity_extractor = EntityExtractor()
    entity_similarity_computer = EntitySimilarityComputer()
    entity_clusterer = EntityCluster(threshold=S, resolution=R)
    cluster_linker = ClusterLinker(threshold=S)
    all_preds = []
    last_tweets = None
    clustering_last = None
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

            processed_tweets_file_name = f'tweets_{"-".join([name.replace(".json", "") for name in subwindow_names])}.pickle'
            if os.path.exists(event_output_dir + processed_tweets_file_name):
                with open(event_output_dir + processed_tweets_file_name, 'rb') as handle:
                    processed_tweets = pickle.load(handle)
            else:
                tweets_df = pd.DataFrame(tweets)
                tweet_ids = tweets_df.tweet_id.tolist()
                processed_tweets = entity_extractor.apply(tweets)

                with open(event_output_dir + processed_tweets_file_name, 'wb') as handle:
                    pickle.dump(processed_tweets, handle, protocol=pickle.HIGHEST_PROTOCOL)

            similarities, entities, tweet_vecs = entity_similarity_computer.apply(processed_tweets)
            partition = entity_clusterer.cluster(similarities)

            start_from = len(set(all_preds))
            clustering_current = partition_to_cluster(partition, start_from=start_from)

            if clustering_last is not None:
                mapping = cluster_linker.link(clustering_last, clustering_current, tweets_last, tweets)

                clustering_current_keys = list(clustering_current.keys())
                for c1, c2 in mapping.items():
                    # print(f'{c2} -> {c1}')
                    clustering_current[list(clustering_last.keys())[c1]] = clustering_current.pop(
                        clustering_current_keys[c2])
                print(mapping)

            clustering_vecs = get_clustering_vectors(clustering_current, len(entities))
            tweet_cluster_sim = cosine_similarity(tweet_vecs, clustering_vecs)
            tweet_clusters = tweet_cluster_sim.argmax(axis=1) + start_from

            all_preds += list(tweet_clusters)
            results_df = pd.DataFrame()
            results_df['tweet_id'] = tweet_ids
            results_df['label'] = tweet_clusters

            clustering_last = clustering_current
            tweets_last = tweets

            if os.path.exists(event_output_dir + 'events.csv'):
                results_df.to_csv(event_output_dir + 'events.csv', index=False, header=None, mode='a', encoding='utf-8')
            else:
                if not os.path.exists(event_output_dir):
                    os.makedirs(event_output_dir)
                results_df.to_csv(event_output_dir + 'events.csv', index=False)

        stop = timeit.default_timer()
        print('Time in minutes: ', (stop - start) / 60)
        current_date = current_date + datetime.timedelta(days=1)


if __name__ == '__main__':
    main()
