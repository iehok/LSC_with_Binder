import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer

from src.z_config import Config
from src.z_utils import set_target_words

config = Config()


def main(args):
    with open(os.path.join(config.DATA_DIR, 'word_ratings/binder_features.txt'), 'r') as f:
        binder_features = f.read().splitlines()

    target_decade1, target_decade2 = list(map(int, args.target_decades.split('-')))

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)

    target_words = set_target_words('wordnet')

    target_ids = tokenizer.convert_tokens_to_ids(target_words)

    target_words = [word for word, id in zip(target_words, target_ids) if id != 100]
    target_ids = [id for id in target_ids if id != 100]

    print(len(target_words), end='')
    idx = 0
    differences = []
    while idx < len(target_words):
        word = target_words[idx]
        decade_to_binder_path = os.path.join(config.COHA_CENTROIDS_DIR, f'wordnet_words/{args.embedding_model}/binder/{word}.json')
        decade_to_binder = pd.read_json(decade_to_binder_path, typ='series').to_dict()
        if (target_decade1 in decade_to_binder) and (target_decade2 in decade_to_binder):
            difference = np.array(decade_to_binder[target_decade2]) - np.array(decade_to_binder[target_decade1])
            differences.append(difference)
            idx += 1
        else:
            target_words.pop(idx)
    differences = np.stack(differences)
    print(' -> ', end='')
    print(len(target_words))

    # -*-*-*-*-*-*-*-
    cook_and_stevenson_words = set(['adornment', 'dynamic', 'skillful', 'synthesis'])  # from Cook & Stevenson (2010)
    haslam_words = set(['abuse', 'addiction', 'bullying', 'prejudice', 'trauma'])  # from Haslam (2016)
    altakhaineh_words = set(['abominable', 'abysmal', 'abhorrent', 'appalling', 'horrible', 'ghastly', 'horrendous', 'terrible'])  # from Altakhaineh (2018)
    vylomova_etal_words = set(['addiction', 'bullying', 'harassment', 'prejudice', 'trauma'])  # from Vylomova et al. (2019)
    wit_words = set(['awful'])  # from de Wit (2021)
    pejoation_words = cook_and_stevenson_words | haslam_words | altakhaineh_words | vylomova_etal_words | wit_words
    print(len(pejoation_words))

    negative_features = ['Pain', 'Harm', 'Unpleasant', 'Sad', 'Angry', 'Disgusted', 'Fearful']
    negative_features_indices = [binder_features.index(feature) for feature in negative_features]

    differences_neg = differences[:, negative_features_indices]

    differences_neg_max = np.max(differences_neg, axis=1)
    # differences_neg_max = np.mean(differences_neg, axis=1)

    differences_neg_max_sorted_indices = np.argsort(differences_neg_max)[::-1]

    x_y_s_list = []
    for i, idx in enumerate(differences_neg_max_sorted_indices):
        word = target_words[idx]
        difference_neg_max = differences_neg_max[idx]
        if word in pejoation_words:
            x_y_s_list.append((i + 1, difference_neg_max, word))
        else:
            x_y_s_list.append((i + 1, difference_neg_max, ''))

    sns.set_style('whitegrid')
    plt.figure(figsize=(6, 4))
    plt.plot([x_y_s[0] for x_y_s in x_y_s_list], [x_y_s[1] for x_y_s in x_y_s_list], color='tab:red')
    cnt = 0
    for x_y_s in x_y_s_list:
        x, y, s = x_y_s
        if s:
            plt.text(x, 0.5 - cnt * 0.4, s, ha='center', va='top')
            plt.vlines(x, 0.5 - cnt * 0.4, y, color='tab:gray', linestyles='dotted')
            cnt += 1
    plt.xlim(left=-1000)
    plt.ylim((-3.3, 3.3))
    plt.xlabel('Rank')
    plt.ylabel('LSC score for negative features')
    plt.savefig('./figures/change_scores_neg.png', dpi=300, bbox_inches='tight')
    # -*-*-*-*-*-*-*-

    # -*-*-*-*-*-*-*-
    cook_and_stevenson_words = set(['disputable', 'hysteria', 'slothful', 'thoughtfulness'])  # from Cook & Stevenson (2010)
    altakhaineh_words = set(['brilliant', 'fabulous', 'fantastic', 'awesome', 'magnificent', 'spectacular', 'marvellous'])  # from Altakhaineh (2018)
    wit_words = set(['terrific', 'awesome'])  # from de Wit (2021)
    amelioration_words = cook_and_stevenson_words | altakhaineh_words | wit_words
    print(len(amelioration_words))

    positive_features = ['Pleasant', 'Happy']
    positive_features_indices = [binder_features.index(feature) for feature in positive_features]

    differences_pos = differences[:, positive_features_indices]

    differences_pos_max = np.max(differences_pos, axis=1)
    # differences_pos_max = np.mean(differences_pos, axis=1)

    differences_pos_max_sorted_indices = np.argsort(differences_pos_max)[::-1]

    x_y_s_list = []
    for i, idx in enumerate(differences_pos_max_sorted_indices):
        word = target_words[idx]
        difference_pos_max = differences_pos_max[idx]
        if word in amelioration_words:
            x_y_s_list.append((i + 1, difference_pos_max, word))
        else:
            x_y_s_list.append((i + 1, difference_pos_max, ''))

    sns.set_style('whitegrid')
    plt.figure(figsize=(6, 4))
    plt.plot([x_y_s[0] for x_y_s in x_y_s_list], [x_y_s[1] for x_y_s in x_y_s_list], color='tab:green')
    cnt = 0
    for x_y_s in x_y_s_list:
        x, y, s = x_y_s
        if s:
            plt.text(x, 0 - cnt * 0.4, s, ha='center', va='top')
            plt.vlines(x, 0 - cnt * 0.4, y, color='tab:gray', linestyles='dotted')
            cnt += 1
    plt.xlim(left=-1000)
    plt.ylim((-3.3, 3.3))
    plt.xlabel('Rank')
    plt.ylabel('LSC score for positive features')
    plt.savefig('./figures/change_scores_pos.png', dpi=300, bbox_inches='tight')
    # -*-*-*-*-*-*-*-


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    parser.add_argument('--embedding_model', type=str, required=True)
    parser.add_argument('--target_decades', type=str, required=True)
    args = parser.parse_args()
    main(args)

# python -B -m src.e_analyze_pejoration_and_amelioration --embedding_model bert-base-uncased --target_decades 1910-2000
