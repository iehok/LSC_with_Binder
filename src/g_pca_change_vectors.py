import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA, SparsePCA
from tqdm import tqdm
from transformers import AutoTokenizer

from src.z_config import Config
from src.z_plot_binder import plot_binder_66
from src.z_utils import set_target_words

config = Config()


def pca_fit_transform(method, n_components, X):
    method_to_class = {
        'PCA': PCA,
        'SparsePCA': SparsePCA,
    }
    pca = method_to_class[method](n_components=n_components, random_state=1)
    X_pca = pca.fit_transform(X)
    return pca, X_pca


def plot_heatmap(components, columns, path):
    df_components = pd.DataFrame(components,
                                 columns=columns,
                                 index=[f'PC{i+1}' for i in range(components.shape[0])])
    plt.figure(figsize=(18, 3))
    sns.heatmap(df_components, cmap='coolwarm', center=0, cbar_kws={"pad": 0.01})
    plt.ylabel('Principal Component')
    plt.xlabel('Feature')
    plt.savefig(path, dpi=300, bbox_inches='tight')


def plot_cumulative_explained_variance(variances, path):
    pseudo_explained_variance_ratio = variances / np.sum(variances, axis=0)
    cumulative_explained_variance = np.cumsum(pseudo_explained_variance_ratio)
    sns.set_style('whitegrid')
    plt.figure(figsize=(4, 2))
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
    plt.xlabel('Principal Component')
    plt.ylabel('CEVR')
    plt.xticks(range(1, len(cumulative_explained_variance) + 1))
    plt.savefig(path, dpi=300, bbox_inches='tight')


def plot_components(X, words, dir):
    for i in range(0, X.shape[1], 2):
        sns.set_style('white')
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, i], X[:, i + 1], s=1)
        for x, w in zip(X, words):
            plt.text(x[i], x[i + 1], w, fontsize=1)
        plt.xlabel(f'Principal Component {i+1}')
        plt.ylabel(f'Principal Component {i+2}')
        plt.savefig(os.path.join(dir, f'pca_{i+1}_{i+2}.png'), dpi=1200, bbox_inches='tight')


def main(args):
    with open(os.path.join(config.DATA_DIR, 'word_ratings/binder_features.txt'), 'r') as f:
        binder_features = f.read().splitlines()

    target_decade1, target_decade2 = list(map(int, args.target_decades.split('-')))

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)

    target_words = set_target_words('wordnet')

    target_ids = tokenizer.convert_tokens_to_ids(target_words)

    target_words = [word for word, id in zip(target_words, target_ids) if id != 100]
    target_ids = [id for id in target_ids if id != 100]

    differences = []
    for word in tqdm(target_words):
        decade_to_binder_path = os.path.join(config.COHA_CENTROIDS_DIR, f'wordnet_words/{args.embedding_model}/binder/{word}.json')
        decade_to_binder = pd.read_json(decade_to_binder_path, typ='series').to_dict()
        difference = np.zeros(65)
        if (target_decade1 in decade_to_binder) and (target_decade2 in decade_to_binder):
            difference = np.array(decade_to_binder[target_decade2]) - np.array(decade_to_binder[target_decade1])
        differences.append(difference)
    differences = np.stack(differences)

    norms = np.linalg.norm(differences, ord=2, axis=1)
    norms_topn_indices = np.argsort(norms)[::-1][:args.top_n]
    differences_topn = differences[norms_topn_indices]
    words_topn = [target_words[i] for i in norms_topn_indices]

    method = 'SparsePCA'
    n_components = 10

    pca, differences_pca = pca_fit_transform(method, n_components, differences_topn)

    variances = np.var(differences_pca, axis=0)
    variances_sorted_indices = np.argsort(variances)[::-1]
    variances_sorted = variances[variances_sorted_indices]
    differences_pca_sorted = differences_pca[:, variances_sorted_indices]
    components_sorted = pca.components_[variances_sorted_indices]

    # plot_heatmap(components_sorted, binder_features, './figures/pca.png')

    print(variances_sorted)
    plot_cumulative_explained_variance(variances_sorted, 'sample.png')

    # plot_components(differences_pca_sorted, words_topn, './figures/')

    # strong_features = np.argsort(components_sorted, axis=1)
    # for i in range(n_components):
    #     print(f'PC{i+1}: ', end='')
    #     for j in range(3):
    #         print(binder_features[strong_features[i][-(j + 1)]], end=' ')
    #     print()

    # components_max = np.sum(np.abs(components_sorted), axis=0)
    # components_max_sorted_indices = np.argsort(components_max)
    # for idx in components_max_sorted_indices:
    #     print(binder_features[idx], end=' ')
    # print()

    # plot_words = ['plane', 'terrific']
    # plot_words_indices = [words_topn.index(w) for w in plot_words]
    # word_and_diff = [{'word': w, 'rep': diff} for w, diff in zip(plot_words, differences_topn[plot_words_indices])]
    # plot_binder_66(word_and_diff, 'sample.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    parser.add_argument('--embedding_model', type=str, required=True)
    parser.add_argument('--target_decades', type=str, required=True)
    parser.add_argument('--top_n', type=int, required=True)
    args = parser.parse_args()
    main(args)

# python -B -m src.e_pca_change_vectors --embedding_model bert-base-uncased --target_decades 1910-2000 --top_n 500
