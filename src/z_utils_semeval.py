import os
import pickle
import sys
from collections import defaultdict
from glob import glob

import torch
from tqdm import tqdm


def load_target_words(path):
    f = open(path, 'r')
    lines = f.read().splitlines()
    target_words = list(map(lambda x: x.split('_')[0], lines))
    return target_words


def encode_corpus(path, corpus, tokenizer):
    if os.path.isfile(path):
        with open(path, mode='br') as f:
            corpus_encoded = pickle.load(f)
    else:
        corpus_encoded = [tokenizer.encode(line, add_special_tokens=False) for line in tqdm(corpus)]
        with open(path, mode='wb') as f:
            pickle.dump(corpus_encoded, f)
    return corpus_encoded


def get_embeddings(dir, corpus_encoded, target_ids, tokenizer, model, device):
    word_to_embeddings = defaultdict(list)

    if glob(os.path.join(dir, '*')):
        for id in target_ids:
            word = tokenizer.convert_ids_to_tokens(id)
            file_path = os.path.join(dir, f'{word}.pt')
            if os.path.isfile(file_path):
                embeddings = torch.load(file_path, weights_only=True)
                word_to_embeddings[word] = embeddings
    else:
        target_ids_set = set(target_ids)
        for ids in tqdm(corpus_encoded):
            word_and_position_tupples = []
            for i, id in enumerate(ids):
                if id in target_ids_set:
                    word = tokenizer.convert_ids_to_tokens(id)
                    position = i + 1
                    word_and_position_tupples.append((word, position))

            if word_and_position_tupples:
                input_ids = [[101] + ids + [102]]
                input_ids = torch.tensor(input_ids)
                input_ids = input_ids.to(device)

                with torch.no_grad():
                    last_hidden_state = model(input_ids).last_hidden_state
                    last_hidden_state = last_hidden_state.cpu()

                for word, position in word_and_position_tupples:
                    embedding = last_hidden_state[0, position]
                    word_to_embeddings[word].append(embedding)

        for word, embeddings in word_to_embeddings.items():
            word_to_embeddings[word] = torch.stack(embeddings)

        for word, embeddings in word_to_embeddings.items():
            file_path = os.path.join(dir, f'{word}.pt')
            torch.save(embeddings, file_path)

    return word_to_embeddings
