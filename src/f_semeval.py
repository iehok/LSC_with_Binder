import argparse
import os
import sys
from pathlib import Path

import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.z_config import Config
from src.z_utils import get_model_class, load_args
from src.z_utils_semeval import encode_corpus, get_embeddings, load_target_words

config = Config()


def main(args):
    device = torch.device('cuda:0')

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)

    embedding_model = AutoModel.from_pretrained(args.embedding_model)
    embedding_model = embedding_model.to(device)
    embedding_model.eval()

    if args.projector_model != 'default':
        ckpt_args = load_args(os.path.join(config.EXP_DIR, f'{args.projector_model}/args.yaml'))
        projector_model = get_model_class(ckpt_args.projector_model)()
        checkpoint = torch.load(ckpt_args.best_ckpt_path, map_location='cpu')
        states = checkpoint['states']
        projector_model.load_state_dict(states)
        projector_model = projector_model.to(device)
        projector_model.eval()

    target_words_path = os.path.join(config.SEMEVAL_DIR, 'targets.txt')
    target_words = load_target_words(target_words_path)
    print(f'target_words:\n{target_words}')

    target_ids = tokenizer.convert_tokens_to_ids(target_words)
    target_ids = [id for id in target_ids if id != 100]

    ccoha1_path = os.path.join(config.SEMEVAL_DIR, 'corpus1/token/ccoha1.txt')
    with open(ccoha1_path, 'r') as f:
        ccoha1 = f.read().splitlines()

    ccoha2_path = os.path.join(config.SEMEVAL_DIR, 'corpus2/token/ccoha2.txt')
    with open(ccoha2_path, 'r') as f:
        ccoha2 = f.read().splitlines()

    ccoha1_encoded_path = os.path.join(config.SEMEVAL_DIR, 'corpus1/token/ccoha1_encoded.pickle')
    ccoha1_encoded = encode_corpus(ccoha1_encoded_path, ccoha1, tokenizer)

    ccoha2_encoded_path = os.path.join(config.SEMEVAL_DIR, 'corpus2/token/ccoha2_encoded.pickle')
    ccoha2_encoded = encode_corpus(ccoha2_encoded_path, ccoha2, tokenizer)

    embeddings1_dir = os.path.join(config.SEMEVAL_DIR, f'corpus1/token/embeddings/{args.embedding_model}')
    word_to_embeddings1 = get_embeddings(embeddings1_dir, ccoha1_encoded, target_ids, tokenizer, embedding_model, device)

    embeddings2_dir = os.path.join(config.SEMEVAL_DIR, f'corpus2/token/embeddings/{args.embedding_model}')
    word_to_embeddings2 = get_embeddings(embeddings2_dir, ccoha2_encoded, target_ids, tokenizer, embedding_model, device)

    if args.projector_model != 'default':
        word_to_embeddings1 = {
            word: projector_model.forward_rep(embeddings.to(device), device).cpu()
            for word, embeddings in word_to_embeddings1.items()
        }
        word_to_embeddings2 = {
            word: projector_model.forward_rep(embeddings.to(device), device).cpu()
            for word, embeddings in word_to_embeddings2.items()
        }
        for word, embeddings1 in word_to_embeddings1.items():
            embeddings1_dir = os.path.join(config.SEMEVAL_DIR, f'corpus1/token/embeddings/{args.projector_model}')
            Path(embeddings1_dir).mkdir(parents=True, exist_ok=True)
            embeddings1_path = os.path.join(embeddings1_dir, f'{word}.pt')
            if not os.path.isfile(embeddings1_path):
                torch.save(embeddings1, embeddings1_path)
        for word, embeddings2 in word_to_embeddings2.items():
            embeddings2_dir = os.path.join(config.SEMEVAL_DIR, f'corpus2/token/embeddings/{args.projector_model}')
            Path(embeddings2_dir).mkdir(parents=True, exist_ok=True)
            embeddings2_path = os.path.join(embeddings2_dir, f'{word}.pt')
            if not os.path.isfile(embeddings2_path):
                torch.save(embeddings2, embeddings2_path)

    with open(os.path.join(config.SEMEVAL_DIR, 'truth/graded.txt'), 'r') as f:
        lines = f.read().splitlines()
    golds = [float(line.split('\t')[1]) for line in lines]

    for metric in ['euclid', 'cosine', 'spearman']:
        apd_all = []
        for word in tqdm(word_to_embeddings1):
            embeddings1 = word_to_embeddings1[word]
            embeddings2 = word_to_embeddings2[word]

            embeddings1 = embeddings1.half()  # from float32 to float16
            embeddings2 = embeddings2.half()  # from float32 to float16

            if metric == 'euclid':
                dist = torch.cdist(embeddings1.unsqueeze(0), embeddings2.unsqueeze(1))
            elif metric == 'cosine':
                sims = torch.nn.functional.cosine_similarity(embeddings1.unsqueeze(0), embeddings2.unsqueeze(1), dim=-1)
                dist = torch.ones(sims.shape) - sims
            elif metric == 'spearman':
                sims = spearmanr(embeddings1, embeddings2, axis=1)[0][:embeddings1.size(0), embeddings2.size(0):]
                dist = torch.ones(sims.shape) - sims

            average_pairwise_distance = torch.mean(dist).item()
            apd_all.append(average_pairwise_distance)

        dist_ave = sum(apd_all) / len(apd_all)
        apd_all = apd_all[:6] + [dist_ave] + apd_all[6:11] + [dist_ave] + apd_all[11:]

        print(f'spearman_r when the metric is "{metric}": {spearmanr(apd_all, golds)[0]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    parser.add_argument('--embedding_model', type=str, required=True)
    parser.add_argument('--projector_model', type=str, default='default')
    args = parser.parse_args()
    main(args)

# python -B -m src.f_semeval --embedding_model bert-base-uncased
# python -B -m src.f_semeval --embedding_model bert-base-uncased --projector_model projector-model=linear-span=19102000
