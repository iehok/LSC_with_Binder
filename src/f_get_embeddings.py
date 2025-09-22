import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.z_config import Config
from src.z_utils import encode_coha, get_context, get_model_class, load_args, load_coha, set_target_words

config = Config()


def main(args):
    device = torch.device('cuda:0')

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)

    embedding_model = AutoModel.from_pretrained(args.embedding_model)
    embedding_model = embedding_model.to(device)
    embedding_model.eval()

    ckpt_args = load_args(os.path.join(config.EXP_DIR, f'{args.projector_model}/args.yaml'))
    projector_model = get_model_class(ckpt_args.model_name)()
    checkpoint = torch.load(ckpt_args.best_ckpt_path, map_location='cpu')
    states = checkpoint['states']
    projector_model.load_state_dict(states)
    projector_model = projector_model.to(device)
    projector_model.eval()

    genre_to_decade_to_lines = load_coha()
    genre_to_decade_to_lines_encoded = encode_coha(genre_to_decade_to_lines, tokenizer)

    target_words = set_target_words(args.target_words)
    target_decades = list(map(int, args.target_decades.split('-')))

    for target_word in tqdm(target_words):
        examples_path = os.path.join(config.COHA_EXAMPLES_DIR, f'{target_word}.jsonl')
        examples = pd.read_json(examples_path, lines=True).to_dict('records')

        original_reps = []
        binder_reps = []
        for i, example in enumerate(examples):
            genre, decade, line, position = example.values()

            if decade not in target_decades:
                continue

            token_ids = genre_to_decade_to_lines_encoded[genre][decade][line]
            context_ids, pos_in_context = get_context(token_ids, position)
            pos_in_context += 1
            input_ids = [[101] + context_ids + [102]]
            input_ids = torch.tensor(input_ids)
            input_ids = input_ids.to(device)

            with torch.no_grad():
                original_rep = embedding_model(input_ids).last_hidden_state[0][pos_in_context]
                binder_rep = projector_model.forward_rep(original_rep, device)

            original_rep = original_rep.cpu().tolist()
            binder_rep = binder_rep.cpu().tolist()

            original_reps.append({'idx': i, 'rep': original_rep})
            binder_reps.append({'idx': i, 'rep': binder_rep})

        df = pd.Series(original_reps)
        df.to_json(os.path.join(args.original_reps_dir, f'{target_word}.jsonl'), force_ascii=False, lines=True, orient='records')

        df = pd.Series(binder_reps)
        df.to_json(os.path.join(args.binder_reps_dir, f'{target_word}.jsonl'), force_ascii=False, lines=True, orient='records')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    parser.add_argument('--embedding_model', type=str, required=True)
    parser.add_argument('--projector_model', type=str, required=True)
    parser.add_argument('--target_decades', type=str, required=True)
    parser.add_argument('--target_words', type=str, required=True)

    args = parser.parse_args()

    args.embeddings_dir = os.path.join(config.COHA_EMBEDDINGS_DIR, f'{args.embedding_model}')

    args.original_reps_dir = os.path.join(args.embeddings_dir, 'original')
    Path(args.original_reps_dir).mkdir(parents=True, exist_ok=True)

    args.binder_reps_dir = os.path.join(args.embeddings_dir, 'binder')
    Path(args.binder_reps_dir).mkdir(parents=True, exist_ok=True)

    main(args)

# python -B -m src.f_get_embeddings --embedding_model bert-base-uncased --projector_model projector-model=linear-span=19602000 --target_decades 1910-2000 --target_words pc
