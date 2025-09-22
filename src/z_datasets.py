import argparse
import os
import random
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast

from src.z_config import Config

config = Config()


class TrainData(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_data(self):
        binder_data = pd.read_json(os.path.join(config.DATA_DIR, 'word_ratings/binder_data.jsonl'), lines=True).to_dict('records')
        binder_word_to_rep = {d['word']: torch.tensor(d['rep']) for d in binder_data}

        word_to_decade_to_freq = pd.read_json(os.path.join(config.COHA_CENTROIDS_DIR, f'word_to_decade_to_freq.json'), typ='series').to_dict()

        target_words = list(word_to_decade_to_freq.keys())

        if self.args.method in ['projector_fold']:
            words_splitted_path = os.path.join(self.args.experiment_dir, 'words_splitted.jsonl')
            words_splitted = pd.read_json(words_splitted_path, lines=True).to_dict('records')
            target_words = words_splitted[self.args.fold]['train']
        elif self.args.method in ['projector']:
            pass
        else:
            print('The model is not implemented.')
            exit()

        target_span_start, target_span_end = list(map(int, self.args.target_span.split('-')))

        word_to_centroid = {}
        for w in tqdm(target_words):
            decade_to_freq = word_to_decade_to_freq[w]
            decade_to_centroid = pd.read_json(os.path.join(config.COHA_CENTROIDS_DIR, f'binder_words/{self.args.embedding_model}/original/{w}.json'), typ='series').to_dict()

            sum_all = []
            for decade in range(target_span_start, target_span_end + 1, 10):
                if decade in decade_to_centroid:
                    centroid = torch.tensor(decade_to_centroid[decade])
                    freq = decade_to_freq[str(decade)]
                    sum_ = centroid * freq
                    sum_all.append(sum_)
            sum_all = torch.stack(sum_all)
            centroid = torch.sum(sum_all, dim=0) / sum(list(decade_to_freq.values()))
            word_to_centroid[w] = centroid

        data = [{'word': w, 'bert': word_to_centroid[w], 'binder': binder_word_to_rep[w]} for w in target_words]
        return data

    def collate_fn(self, batch):
        return {
            'bert': torch.stack([b['bert'] for b in batch]),
            'binder': torch.stack([b['binder'] for b in batch]),
        }


class DevData(Dataset):
    def __init__(self, args, task):
        super().__init__()
        self.args = args
        self.task = task
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_data(self):
        binder_data = pd.read_json(os.path.join(config.DATA_DIR, 'word_ratings/binder_data.jsonl'), lines=True).to_dict('records')
        binder_word_to_rep = {d['word']: torch.tensor(d['rep']) for d in binder_data}

        word_to_decade_to_freq = pd.read_json(os.path.join(config.COHA_CENTROIDS_DIR, f'word_to_decade_to_freq.json'), typ='series').to_dict()

        target_words = list(word_to_decade_to_freq.keys())

        if self.args.method in ['projector_fold']:
            words_splitted_path = os.path.join(self.args.experiment_dir, 'words_splitted.jsonl')
            words_splitted = pd.read_json(words_splitted_path, lines=True).to_dict('records')
            target_words = words_splitted[self.args.fold]['val']
        elif self.args.method in ['projector']:
            target_words = []
        else:
            print('The model is not implemented.')
            exit()

        target_span_start, target_span_end = list(map(int, self.args.target_span.split('-')))

        word_to_centroid = {}
        for w in tqdm(target_words):
            decade_to_freq = word_to_decade_to_freq[w]
            decade_to_centroid = pd.read_json(os.path.join(config.COHA_CENTROIDS_DIR, f'binder_words/{self.args.embedding_model}/original/{w}.json'), typ='series').to_dict()

            sum_all = []
            for decade in range(target_span_start, target_span_end + 1, 10):
                if decade in decade_to_centroid:
                    centroid = torch.tensor(decade_to_centroid[decade])
                    freq = decade_to_freq[str(decade)]
                    sum_ = centroid * freq
                    sum_all.append(sum_)
            sum_all = torch.stack(sum_all)
            centroid = torch.sum(sum_all, dim=0) / sum(list(decade_to_freq.values()))
            word_to_centroid[w] = centroid

        data = [{'word': w, 'bert': word_to_centroid[w], 'binder': binder_word_to_rep[w]} for w in target_words]
        return data

    def collate_fn(self, batch):
        assert len(batch) == 1
        return batch[0]


class TestData(Dataset):
    def __init__(self, args, task):
        super().__init__()
        self.args = args
        self.task = task
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.tokenizer_kwargs = {
            'max_length': config.max_length,
            'padding': 'max_length',
            'truncation': True,
            'return_offsets_mapping': True,
            'return_tensors': 'pt',
        }
        self.data = self.load_data()
        self.train_data = TrainData(args)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_data(self):
        data_path_map = {
            'coha': os.path.join(config.COHA_PREPARED_DIR, f'{self.args.target_stem}.jsonl'),
            'wsd': os.path.join(config.WSD_PREPARED_DIR, 'test/test.jsonl'),
        }
        data = pd.read_json(data_path_map[self.task], lines=True).to_dict('records')
        return data

    def collate_fn(self, batch):
        assert len(batch) == 1
        example = batch[0]

        # about query set
        q_text, q_targetword, q_span = example['text'], example['stem'], example['span']
        q_encoded = self.tokenizer.encode_plus(q_text, **self.tokenizer_kwargs)
        query_ids = q_encoded['input_ids']

        # about support set
        support_sensekeys = list(self.train_data.word_to_sensekeys[q_targetword])
        support_ids, support_spans = [], []
        for s_key in support_sensekeys:
            examples = self.train_data.sensekey_to_examples[s_key]
            random.shuffle(examples)
            examples = examples[:self.args.max_inference_supports]
            s_texts, _, _, _, _, _, s_spans = list(map(list, zip(*[example.values() for example in examples])))
            support_encoded = self.tokenizer.batch_encode_plus(s_texts, **self.tokenizer_kwargs)
            support_ids.append(support_encoded['input_ids'])
            support_spans.append(s_spans)

        return {
            'example': example,
            'query_ids': query_ids,
            'query_spans': [q_span],
            'support_ids': support_ids,
            'support_spans': support_spans,
            'support_sensekeys': support_sensekeys,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0], conflict_handler="resolve")
    parser.add_argument("--max_inference_supports", type=int, default=30)
    parser.add_argument("--method", type=str, default="projector")
    parser.add_argument("--ns", type=int, default=5)
    parser.add_argument("--nq", type=int, default=20)
    args = parser.parse_args()

    data = TrainData(args)
    print(data[0])
