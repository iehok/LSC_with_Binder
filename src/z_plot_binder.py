import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.z_config import Config

config = Config()

f = open(os.path.join(config.DATA_DIR, 'word_ratings/binder_features.txt'), 'r')
binder_features = f.read().splitlines()


def plot_binder_06(data, path):
    assert len(data) <= 2
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(16, 2))
    for i, d in enumerate(data):
        colors = ['tab:blue', 'tab:red']
        markers = ['o', '^']
        plt.plot([i for i in range(65)], d['rep'], color=colors[i], marker=markers[i], markersize=5, label=d['word'])
    plt.xticks([i for i in range(65)], labels=binder_features, rotation=90)
    plt.yticks([i for i in range(0, 7, 2)])
    plt.xlim((-1, 65))
    plt.ylim((-0.3, 6.3))
    plt.legend(loc='upper right', ncols=len(data))
    plt.savefig(path, dpi=300, bbox_inches='tight')


def plot_binder_66(data, path):
    assert len(data) <= 2
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(16, 2))
    for i, d in enumerate(data):
        colors = ['tab:blue', 'tab:red']
        markers = ['o', '^']
        plt.plot([i for i in range(65)], d['rep'], color=colors[i], marker=markers[i], markersize=5, label=d['word'])
    plt.xticks([i for i in range(65)], labels=binder_features, rotation=90)
    plt.yticks([i for i in range(-6, 7, 3)])
    plt.xlim((-1, 65))
    plt.ylim((-6.3, 6.3))
    plt.legend(loc='upper right', ncols=len(data))
    plt.savefig(path, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    binder_data = pd.read_json(os.path.join(config.DATA_DIR, 'word_ratings/binder_data.jsonl'), lines=True).to_dict('records')
    plot_binder_06([d for d in binder_data if d['word'] in ['dog', 'coffee']], 'sample1.png')
    plot_binder_66([d for d in binder_data if d['word'] in ['dog', 'coffee']], 'sample2.png')
