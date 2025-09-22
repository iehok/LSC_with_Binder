from pathlib import Path


class Config:
    PROJECT_NAME = 'ENTER_YOUR_PROJECT_NAME'
    ROOT_DIR = 'ENTER_YOUR_ROOT_DIR'
    PROJECT_DIR = ROOT_DIR / PROJECT_NAME
    EXP_DIR = 'ENTER_YOUR_EXP_DIR'
    DATA_DIR = 'ENTER_YOUR_DATA_DIR'

    COHA_ORIGINAL_DIR = PROJECT_DIR / 'data/coha/0_original'
    COHA_ENCODED_DIR = PROJECT_DIR / 'data/coha/1_encoded'
    COHA_EXAMPLES_DIR = DATA_DIR / 'coha/2_examples'
    COHA_CENTROIDS_DIR = DATA_DIR / 'coha/3_centroids'
    COHA_EMBEDDINGS_DIR = DATA_DIR / 'coha/4_embeddings'
    COHA_GENRES = ['acad', 'fic', 'mag', 'news', 'tvm']
    COHA_GENRE_TO_START = {
        'acad': 1820,
        'fic': 1820,
        'mag': 1820,
        'news': 1860,
        'tvm': 1930,
    }
    COHA_START = 1910
    COHA_END = 2000
    COHA_SPAN_SIZE = 10

    SEMEVAL_DIR = DATA_DIR / 'semeval2020_ulscd_eng'

    max_length = 128
