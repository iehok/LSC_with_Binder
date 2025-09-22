import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from transformers import AutoTokenizer

from src.z_config import Config
from src.z_plot import plot_dist_h, plot_embs
from src.z_utils import encode_coha, find_top_n_closest_to_centroids, get_context, get_labels_and_centroids, load_coha, set_target_words

config = Config()


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)

    target_words = set_target_words(args.target_words)
    target_ids = tokenizer.convert_tokens_to_ids(target_words)

    target_words = [word for word, id in zip(target_words, target_ids) if id != 100]
    target_ids = [id for id in target_ids if id != 100]

    genre_to_decade_to_lines = load_coha()
    genre_to_decade_to_lines_encoded = encode_coha(genre_to_decade_to_lines, tokenizer)

    for word in target_words:
        print()
        print(f'target_word = {word}')

        examples_path = os.path.join(config.COHA_EXAMPLES_DIR, f'{word}.jsonl')
        examples = pd.read_json(examples_path, lines=True).to_dict('records')

        decade_to_indeces = defaultdict(list)
        for i, example in enumerate(examples):
            decade_to_indeces[example['decade']].append(i)
        print({k: len(v) for k, v in decade_to_indeces.items()})

        binder_reps_path = os.path.join(config.COHA_EMBEDDINGS_DIR, f'{args.embedding_model}/binder/{word}.jsonl')
        binder_reps = pd.read_json(binder_reps_path, lines=True).to_dict('records')

        examples = [examples[x['idx']] for x in binder_reps]
        reps = np.array([x['rep'] for x in binder_reps])

        labels, centroids = get_labels_and_centroids(reps, 10)

        top_n = 5
        top_n_closest_to_centroids = find_top_n_closest_to_centroids(reps, labels, centroids, top_n)

        for i in range(len(top_n_closest_to_centroids)):
            print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
            print('label:', i)
            for idx in top_n_closest_to_centroids[i]:
                example = examples[idx]
                genre, decade, line, position = example.values()
                token_ids = genre_to_decade_to_lines_encoded[genre][decade][line]
                context_ids, pos_in_context = get_context(token_ids, position)
                context = tokenizer.decode(context_ids)
                print(f'* {context}, {decade}')
            print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')

        # plot_embs(reps, labels, f'./figures/plot_2D-word={word}.png')

        decades = sorted(list(set([example['decade'] for example in examples])))
        n_labels = len(set(labels))
        decade_to_freqs = {decade: [0] * n_labels for decade in decades}
        for label, example in zip(labels, examples):
            decade_to_freqs[example['decade']][label] += 1
        for decade, freqs in decade_to_freqs.items():
            sum_freqs = sum(freqs)
            for i, freq in enumerate(freqs):
                decade_to_freqs[decade][i] = freq / sum_freqs
        # plot_dist_h(decade_to_freqs, f'./figures/dist_{word}.png')

        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
        # embs_2d = TSNE(n_components=2, perplexity=min(len(labels) - 1, 30), random_state=1).fit_transform(reps)
        # sns.set_style('white')
        # fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
        # cmap = plt.get_cmap('tab10')
        # labels_ = ['amazing', 'causing terror']
        # kwargs = {
        #     'alpha': 0.8,
        # }
        # for i, label in enumerate(labels_):
        #     embs = embs_2d[[j for j, l in enumerate(labels) if l == i]]
        #     ax.scatter(embs[:, 0], embs[:, 1], c=cmap(i), **kwargs)
        # ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        # plt.savefig(f'tsne_{word}.png', dpi=300, bbox_inches='tight')

        sense_def_path = os.path.join(config.DATA_DIR, f'coha/sense_def.json')
        sense_def = pd.read_json(sense_def_path, typ='series').to_dict()

        decades = sorted(list(decade_to_freqs.keys()), reverse=True)
        assert len(decades) == 2
        sns.set_style('white')
        plt.figure(figsize=(5, 0.8))  # for Appendix
        # plt.figure(figsize=(5, 1))  # for Fig. 4
        cmap = plt.get_cmap('tab10')
        for i, decade in enumerate(decades):
            dist = decade_to_freqs[decade]
            left = 0
            for j, freq in enumerate(dist):
                plt.barh(i, freq, left=left, color=cmap(j), linewidth=0, label=sense_def.get(word)[j] if i == 0 else None)
                left += freq
        plt.xticks([i * 0.2 for i in range(6)])
        plt.xlim(0, 1)
        plt.yticks([i for i in range(len(decades))], labels=[f'{d}s' for d in decades])
        plt.legend(loc='upper center', ncols=1 if len(sense_def.get(word)) == 2 else 2, bbox_to_anchor=(0.45, 1.8), fontsize=8, columnspacing=0.5, handletextpad=0.2)  # for Appendix
        # plt.legend(loc='upper center', ncols=2, bbox_to_anchor=(0.45, 1.6), fontsize=10, columnspacing=0.5, handletextpad=0.2)  # for Fig. 4
        plt.savefig(f'sense_dist_{word}.png', dpi=300, bbox_inches='tight')
        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    parser.add_argument('--embedding_model', type=str, required=True)
    parser.add_argument('--target_words', type=str, required=True)
    args = parser.parse_args()
    main(args)

# python -B -m src.g_clustering_embeddings --embedding_model bert-base-uncased --target_words pc
