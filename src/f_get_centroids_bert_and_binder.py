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
from z_utils import encode_coha, get_context, get_model_class, load_coha, set_target_words

config = Config()


def main(args):
    device = torch.device('cuda:0')

    tokenizer = BertTokenizer.from_pretrained(args.embedding_model)

    bert = BertModel.from_pretrained(args.embedding_model)
    bert = bert.to(device)
    bert.eval()

    projector = get_model_class(args.regression_model)()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    states = checkpoint['states']
    projector.load_state_dict(states)
    projector = projector.to(device)
    projector.eval()

    genre_to_decade_to_lines = load_coha()
    genre_to_decade_to_lines_encoded = encode_coha(genre_to_decade_to_lines, tokenizer)

    target_words = set_target_words('wordnet')
    target_span_start, target_span_end = list(map(int, args.target_span.split('-')))

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
        decade_to_binder_reps = defaultdict(list)
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
                binder_rep = projector.forward_rep(bert_rep, device)

            bert_rep = bert_rep.cpu()
            binder_rep = binder_rep.cpu()

            decade_to_bert_reps[decade].append(bert_rep)
            decade_to_binder_reps[decade].append(binder_rep)

        decade_to_bert_centroid = {}
        for decade, reps in decade_to_bert_reps.items():
            reps = torch.stack(reps)
            centroid = torch.mean(reps, dim=0)
            centroid = centroid.tolist()
            decade_to_bert_centroid[decade] = centroid

        df = pd.Series(decade_to_bert_centroid)
        df.to_json(os.path.join(args.centroids_dir, f'original/{target_word}.json'))

        decade_to_binder_centroid = {}
        for decade, reps in decade_to_binder_reps.items():
            reps = torch.stack(reps)
            centroid = torch.mean(reps, dim=0)
            centroid = centroid.tolist()
            decade_to_binder_centroid[decade] = centroid

        df = pd.Series(decade_to_binder_centroid)
        df.to_json(os.path.join(args.centroids_dir, f'binder/{target_word}.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--embedding_model', type=str, required=True)
    parser.add_argument('--regression_model', type=str, required=True)
    parser.add_argument('--target_span', type=str, required=True)
    args = parser.parse_args()

    args.centroids_dir = os.path.join(config.COHA_CENTROIDS_DIR, f'wordnet_words/{args.embedding_model}')
    Path(args.centroids_dir).mkdir(parents=True, exist_ok=True)

    main(args)

# python -B -m src.get_centroids_bert_and_binder --embedding_model bert-base-uncased --target_span 1910-2000
