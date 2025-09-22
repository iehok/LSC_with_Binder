import os
import sys

import pandas as pd

from src.z_config import Config

config = Config()


def main():
    data = pd.read_csv(os.path.join(config.DATA_DIR, 'word_ratings/WordSet1_Ratings.csv')).to_dict('records')

    features = list(data[0].keys())[5:70]

    with open(os.path.join(config.DATA_DIR, 'word_ratings/binder_features.txt'), 'w') as f:
        f.write(('\n').join(features))

    data_new = []
    for d in data:
        word = d['Word']
        rep = list(d.values())[5:70]
        rep = [0.0 if x == 'na' else float(x) for x in rep]
        data_new.append({
            'word': word,
            'rep': rep,
        })

    df = pd.DataFrame(data_new)
    df.to_json(os.path.join(config.DATA_DIR, 'word_ratings/binder_data.jsonl'), force_ascii=False, lines=True, orient='records')


if __name__ == '__main__':
    main()
