import os
import sys
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer

from src.z_config import Config
from src.z_utils import encode_coha, load_coha, set_target_words

config = Config()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def main():
    target_words = set_target_words('binder') + set_target_words('wordnet')

    target_ids = tokenizer.convert_tokens_to_ids(target_words)

    target_words = [word for word, id in zip(target_words, target_ids) if id != 100]
    target_ids = [id for id in target_ids if id != 100]

    print(f'target_words: {target_words}')
    print(f'target_ids  : {target_ids}')

    genre_to_decade_to_lines = load_coha()
    genre_to_decade_to_lines_encoded = encode_coha(genre_to_decade_to_lines, tokenizer)

    target_ids_set = set(target_ids)
    word_to_examples = defaultdict(list)
    for genre, decade_to_lines in genre_to_decade_to_lines_encoded.items():
        print(genre)
        for decade, lines in tqdm(decade_to_lines.items()):
            for i, line in enumerate(lines):
                for position, id in enumerate(line):
                    if id in target_ids_set:
                        word = tokenizer.convert_ids_to_tokens(id)
                        word_to_examples[word].append({
                            'genre': genre,
                            'decade': decade,
                            'line': i,
                            'position': position,
                        })

    for word, examples in word_to_examples.items():
        df = pd.DataFrame(examples)
        df.to_json(os.path.join(config.COHA_EXAMPLES_DIR, f'{word}.jsonl'), force_ascii=False, lines=True, orient='records')


if __name__ == '__main__':
    main()
