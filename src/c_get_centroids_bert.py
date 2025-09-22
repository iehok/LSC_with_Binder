import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from z_config import Config
from z_utils import encode_coha, get_context, load_coha, set_target_words

config = Config()


def main(args):
    device = torch.device('cuda:0')

    tokenizer = BertTokenizer.from_pretrained(args.embedding_model)

    bert = BertModel.from_pretrained(args.embedding_model)
    bert = bert.to(device)
    bert.eval()

    genre_to_decade_to_lines = load_coha()
    genre_to_decade_to_lines_encoded = encode_coha(genre_to_decade_to_lines, tokenizer)

    target_words = set_target_words('binder')
    target_span_start, target_span_end = list(map(int, args.target_span.split('-')))

    word_to_decade_to_freq = {}
    for target_word in tqdm(target_words):
        examples_path = os.path.join(config.COHA_EXAMPLES_DIR, f'{target_word}.jsonl')

        if not os.path.isfile(examples_path):
            print(f'{examples_path} does not exist.')
            continue

        examples = pd.read_json(examples_path, lines=True).to_dict('records')

        examples = [
            example for example in examples
            if (example['decade'] >= target_span_start) and (example['decade'] <= target_span_end)
        ]

        decade_to_bert_reps = defaultdict(list)
        for example in examples:
            genre, decade, line, position = example.values()

            token_ids = genre_to_decade_to_lines_encoded[genre][decade][line]
            context_ids, pos_in_context = get_context(token_ids, position)
            pos_in_context += 1
            input_ids = [[101] + context_ids + [102]]
            input_ids = torch.tensor(input_ids)
            input_ids = input_ids.to(device)

            with torch.no_grad():
                bert_rep = bert(input_ids).last_hidden_state[0][pos_in_context]

            bert_rep = bert_rep.cpu()

            decade_to_bert_reps[decade].append(bert_rep)

        decade_to_bert_centroid = {}
        for decade, reps in decade_to_bert_reps.items():
            reps = torch.stack(reps)
            centroid = torch.mean(reps, dim=0)
            centroid = centroid.tolist()
            decade_to_bert_centroid[decade] = centroid

        df = pd.Series(decade_to_bert_centroid)
        df.to_json(os.path.join(args.centroids_dir, f'{target_word}.json'))

        decade_to_freq = defaultdict(int)
        for example in examples:
            genre, decade, line, position = example.values()
            decade_to_freq[decade] += 1
        word_to_decade_to_freq[target_word] = decade_to_freq

    df = pd.Series(word_to_decade_to_freq)
    df.to_json(os.path.join(config.COHA_CENTROIDS_DIR, f'word_to_decade_to_freq.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    parser.add_argument('--embedding_model', type=str, required=True)
    parser.add_argument('--target_span', type=str, required=True)
    args = parser.parse_args()

    args.centroids_dir = os.path.join(config.COHA_CENTROIDS_DIR, f'binder_words/{args.embedding_model}')
    Path(args.centroids_dir).mkdir(parents=True, exist_ok=True)

    main(args)

# python -B -m src.get_centroids_bert --embedding_model bert-base-uncased --target_span 1910-2000
