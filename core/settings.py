import nltk

ABC = 'abcdefghijklmnñopqrstuvwxyzáéíóúü0123456789 '
REGEX_PATTERN = "[^{0}{1}_]".format(ABC, ABC.upper())
SENTENCE_REGEX_PATTERN = "[^{0}{1}_.\n]".format(ABC, ABC.upper())
RANDOM_STATE = 42
N_TOP_DOCS = 15
CARACTERES = 500
DATA_PATH = "data/cc_dump_p.csv"
PP_DATA_PATH = "data/cc_dump_p_pp.csv"
LDA_WORKERS = 4
POOL_SIZE = 8
VERSION = '0.1'
CORPUS_PATH = f"data/corpus/corpus_cc_p_v{VERSION}.txt"
LDA_CORPUS_PATH = "data/cc_lemma_corpus_sample.txt"
MODEL_PATH = f"model/cc_model_{VERSION}.vec"
LDA_DICT_PATH = "model/lda/cc_sample.dict"
LDA_TFIDF_PATH = "model/lda/cc_sample.tfidf"
LDA_MODEL_PATH = "model/lda/cc_sample.lda"
VECTORS_PATH = f"model/vectors/vectors_{VERSION}.h5"
VECTORS_INDEX_PATH = f"model/vectors/vector_indices_{VERSION}.h5"
WV_PATH = "model/wv/cc.wv"
MODEL_DIM = 100
MODEL_WINDOW = 5
MODEL_EPOCHS = 5
P_WINDOW_SIZE = 3
THRESHOLD_SIM = 0.90
LDA_SAMPLE_SIZE = 0.10
LDA_NUM_TOPICS = 10
LDA_DISPLAY_WORDS = 10

ES_STOPWORDS = nltk.corpus.stopwords.words("spanish")
ES_STOPWORDS.remove("no")
ES_STOPWORDS.append("<NUM>")