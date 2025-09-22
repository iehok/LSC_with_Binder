import glob
import os
import pickle
import random
from argparse import Namespace
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from src.z_config import Config
from src.z_models import MLP, Linear

config = Config()


def make_folds(data, path):
    n_folds = 10
    n_val = len(data) // n_folds
    n_train = len(data) - n_val

    random.seed(1)
    random.shuffle(data)

    data_splitted = []
    for i in range(n_folds):
        data_train = data[:n_val * i] + data[n_val * (i + 1):]
        data_val = data[n_val * i:n_val * (i + 1)]
        data_splitted.append({
            'train': data_train,
            'val': data_val,
        })

    df = pd.DataFrame(data_splitted)
    df.to_json(path, force_ascii=False, lines=True, orient='records')


def get_model_class(model_name):
    model_name_to_class = {
        'linear': Linear,
        'mlp': MLP,
    }
    return model_name_to_class[model_name]


def set_target_words(key):
    target_words = []
    if key == 'binder':
        binder_data_path = os.path.join(config.DATA_DIR, 'word_ratings/binder_data.jsonl')
        binder_data = pd.read_json(binder_data_path, lines=True).to_dict('records')
        binder_words = [d['word'] for d in binder_data]
        target_words += binder_words
    elif key == 'wordnet':
        wordnet_words_path = os.path.join(config.DATA_DIR, 'coha/wordnet_words.txt')
        wordnet_words = open(wordnet_words_path, 'r').read().splitlines()
        target_words += wordnet_words
    elif key == 'neighbors':
        neighbors_path = os.path.join(config.DATA_DIR, 'coha/neighbors.txt')
        neighbors = open(neighbors_path, 'r').read().splitlines()
        target_words += neighbors
    elif key == 'pc':
        pc_path = os.path.join(config.DATA_DIR, 'coha/pc.txt')
        pc = open(pc_path, 'r').read().splitlines()
        target_words += pc
    return target_words


def load_coha():
    genre_to_decade_to_lines = {genre: {} for genre in config.COHA_GENRES}

    print('-*-*-*-*-*-*-*-')
    for genre in config.COHA_GENRES:
        print(genre)
        for decade in tqdm(range(config.COHA_START, config.COHA_END + 1, config.COHA_SPAN_SIZE)):
            file_path = os.path.join(config.COHA_ORIGINAL_DIR, f'text_{genre}_{decade}.txt')
            if not os.path.isfile(file_path):
                continue
            f = open(file_path)
            lines = f.readlines()
            lines = lines[1:]  # line[0] is empty
            genre_to_decade_to_lines[genre][decade] = lines
    print('-*-*-*-*-*-*-*-')

    return genre_to_decade_to_lines


def encode_coha(genre_to_decade_to_lines, tokenizer):
    genre_to_decade_to_lines_encoded = {genre: defaultdict(list) for genre in config.COHA_GENRES}

    print('-*-*-*-*-*-*-*-')
    for genre, decade_to_lines in genre_to_decade_to_lines.items():
        print(genre)
        for decade, lines in tqdm(decade_to_lines.items()):
            file_path = os.path.join(config.COHA_ENCODED_DIR, f'{genre}_{decade}.pickle')
            if os.path.isfile(file_path):
                with open(file_path, mode='br') as f:
                    genre_to_decade_to_lines_encoded[genre][decade] = pickle.load(f)
            else:
                for line in lines:
                    line_encoded = tokenizer.encode(line, add_special_tokens=False)
                    genre_to_decade_to_lines_encoded[genre][decade].append(line_encoded)
                with open(file_path, mode='wb') as f:
                    pickle.dump(genre_to_decade_to_lines_encoded[genre][decade], f)
    print('-*-*-*-*-*-*-*-')

    return genre_to_decade_to_lines_encoded


def get_context(token_ids, target_position):
    # -2 as [CLS] and [SEP] tokens will be added later; /2 as it's a one-sided window
    window_size = int((config.max_length - 2) / 2)
    context_start = max([0, target_position - window_size])
    padding_offset = max([0, window_size - target_position])
    padding_offset += max([0, target_position + window_size - len(token_ids)])

    context_ids = token_ids[context_start:target_position + window_size]
    context_ids += padding_offset * [0]

    new_target_position = target_position - context_start

    return context_ids, new_target_position


def save_best_model(pl_module):
    # save arguments to "args.yaml"
    pl_module.args.best_ckpt_path = \
        os.path.join(pl_module.args.experiment_dir, f'global_step={pl_module.global_step}.pt')
    with open(pl_module.args.args_path, 'w') as f:
        yaml.dump(vars(pl_module.args), f, default_flow_style=False)

    # delete other "*.pt" files
    for rm_path in glob.glob(os.path.join(pl_module.args.experiment_dir, '*.pt')):
        os.remove(rm_path)

    # save the model
    checkpoint = {
        'states': pl_module.model.state_dict(),
        'optimizer_states': pl_module.optimizer.state_dict()
    }
    torch.save(checkpoint, pl_module.args.best_ckpt_path)
    print(f'Model saved at: {pl_module.args.best_ckpt_path}')


def get_labels(embs, max_n_clusters):
    best_model = get_best_model(embs, max_n_clusters)
    return best_model.labels_


def get_labels_and_centroids(embs, max_n_clusters):
    best_model = get_best_model(embs, max_n_clusters)
    return best_model.labels_, best_model.cluster_centers_


def get_best_model(embs, max_n_clusters):
    best_model, best_score = None, -1
    for k in tqdm(range(2, min(embs.shape[0], max_n_clusters + 1))):
        kmeans = KMeans(n_clusters=k, random_state=1)
        cluster_labels = kmeans.fit_predict(embs)
        score = silhouette_score(embs, cluster_labels)
        if score > best_score:
            best_model = kmeans
            best_score = score
    return best_model


def find_top_n_closest_to_centroids(embs, labels, centroids, top_n):
    n_clusters = len(centroids)
    closest_items = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_data = embs[cluster_indices]
        distances = np.linalg.norm(cluster_data - centroids[cluster_id], axis=1)
        top_indices = cluster_indices[np.argsort(distances)[:top_n]]
        closest_items.append(top_indices)
    return closest_items


def save_labels(labels, path):
    labels_str = list(map(str, labels))
    f = open(path, 'w')
    f.write(('\n').join(labels_str))


def load_labels(path):
    f = open(path, 'r')
    labels = f.read().splitlines()
    labels = list(map(int, labels))
    return labels


def load_args(path):
    with open(path) as f:
        args = yaml.load(f, Loader=yaml.Loader)
    args = Namespace(**args)
    return args
