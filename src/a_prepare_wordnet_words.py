import os
import sys

from nltk.corpus import wordnet as wn
from transformers import BertTokenizer

from src.z_config import Config

config = Config()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def filtering(lemma_names, pos):
    lemma_names = [lemma_name for lemma_name in lemma_names if len(lemma_name) >= 4]
    lemma_names = [lemma_name for lemma_name in lemma_names if tokenizer.convert_tokens_to_ids(lemma_name) != 100]
    lemma_names = [lemma_name for lemma_name in lemma_names if len(wn.synsets(lemma_name, pos=pos)) >= 2]
    return lemma_names


def main():
    all_lemma_names_n = list(wn.all_lemma_names(pos='n'))
    all_lemma_names_v = list(wn.all_lemma_names(pos='v'))
    all_lemma_names_a = list(wn.all_lemma_names(pos='a'))
    all_lemma_names_r = list(wn.all_lemma_names(pos='r'))

    lemma_names_n = filtering(all_lemma_names_n, 'n')
    lemma_names_v = filtering(all_lemma_names_v, 'v')
    lemma_names_a = filtering(all_lemma_names_a, 'a')
    lemma_names_r = filtering(all_lemma_names_r, 'r')

    lemma_names = list(set(lemma_names_n + lemma_names_v + lemma_names_a + lemma_names_r))
    lemma_names = sorted(lemma_names)

    output_path = os.path.join(config.DATA_DIR, 'coha/wordnet_words.txt')
    with open(output_path, 'w') as f:
        f.write(('\n').join(lemma_names))


if __name__ == '__main__':
    main()
