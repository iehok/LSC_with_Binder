import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from src.z_config import Config

config = Config()


def plot_dist_v(decade_to_freqs, path):
    n_decades = len(decade_to_freqs)

    sns.set_style('white')
    fig, ax = plt.subplots(1, 1, figsize=(0.6 * n_decades, 3), dpi=300)

    cmap = plt.get_cmap('tab10')
    for i, freqs in enumerate(decade_to_freqs.values()):
        bottom = 0
        for j, freq in enumerate(freqs):
            ax.bar(i, freq, bottom=bottom, color=cmap(j), linewidth=0)
            bottom += freq

    ax.set_xticks([i for i in range(n_decades)])
    ax.set_xticklabels(list(decade_to_freqs.keys()))
    ax.set_yticks([i * 0.2 for i in range(6)])
    ax.set_ylim(0, 1)
    plt.savefig(path, dpi=300, bbox_inches='tight')


def plot_dist_h(decade_to_dist, path):
    decades = sorted(list(decade_to_dist.keys()), reverse=True)
    n_decades = len(decades)

    sns.set_style('white')
    fig, ax = plt.subplots(1, 1, figsize=(3, 0.3 * n_decades), dpi=300)
    cmap = plt.get_cmap('tab10')

    for i, decade in enumerate(decades):
        dist = decade_to_dist[decade]
        left = 0
        for j, freq in enumerate(dist):
            ax.barh(i, freq, left=left, color=cmap(j), linewidth=0)
            left += freq

    ax.set_xticks([i * 0.2 for i in range(6)])
    ax.set_xlim(0, 1)
    ax.set_yticks([i for i in range(n_decades)])
    ax.set_yticklabels(decades)
    plt.savefig(path, dpi=300, bbox_inches='tight')


def plot_embs(embs, labels, path):
    embs_2d = TSNE(n_components=2, perplexity=min(len(labels) - 1, 30), random_state=1).fit_transform(embs)

    sns.set_style('white')
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)

    cmap = plt.get_cmap('tab10')
    kwargs = {
        'alpha': 0.5,
    }
    for emb, label in zip(embs_2d, labels):
        ax.scatter(emb[0], emb[1], c=cmap(label), **kwargs)
    ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.savefig(path, dpi=300, bbox_inches='tight')


def plot_embs_and_texts(embs, labels, texts, path):
    embs_2d = TSNE(n_components=2, random_state=1).fit_transform(embs)

    sns.set_style('white')
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)

    cmap = plt.get_cmap('tab10')
    kwargs_scatter = {
        'alpha': 0.5,
    }
    kwargs_text = {
        'fontsize': 3,
    }
    for emb, label, text in zip(embs_2d, labels, texts):
        ax.scatter(emb[0], emb[1], c=cmap(label), **kwargs_scatter)
        ax.text(emb[0], emb[1], s=text, **kwargs_text)
    ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.savefig(path, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    decade_to_dist = {1910: [0.1, 0.9], 2000: [0.4, 0.6]}
    plot_dist_h(decade_to_dist, './sample.png')
