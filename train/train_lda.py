#!/usr/bin/env python
# coding: utf-8

import nltk
import logging

import pandas as pd
import numpy as np

import os
import random
import spacy
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.ldamulticore import LdaMulticore
from gensim import corpora, models

from sklearn.feature_extraction.text import CountVectorizer

from core.settings import *
from core.preprocess import *


# diccionarios especiales para puntuación y palabras vacias
nltk.download('punkt') # Manejo de puntuación
nltk.download('wordnet')
nltk.download('stopwords')

nlp = spacy.load("es_core_news_md")

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class CorpusIterator:
    def __init__(self, file_path: str, dictionary):
        self.dictionary = dictionary
        self.file_path = file_path
    def __iter__(self):
        for line in open(self.file_path):
            # assume there's one document per line, tokens separated by whitespace
            yield self.dictionary.doc2bow(line.lower().strip().split())


class TfidfCorpus:
    def __init__(self, corpus, model):
        self.corpus = corpus
        self.model = model
    def __iter__(self):
        for line in self.corpus:
            # assume there's one document per line, tokens separated by whitespace
            yield self.model[line]


def train_lda(dataset_path:str = None, print_stats:bool = True, save_pp_file:bool= True, pp_file_path:str = None, run_preprocess:bool=True):

    """
    Train search engine
    """
    if dataset_path is None:    
        dataset_path = DATA_PATH
    
    if pp_file_path is None:
        pp_file_path = PP_DATA_PATH
    
    if not os.path.isfile(pp_file_path):
        run_preprocess = True
    
    df = pd.read_csv(dataset_path)
    df.rename({"class":"class_", "name":"sentencia"}, axis=1, inplace=True)

    df = df[[isinstance(x, str) for x in df.text.values]]

    # ## **Crear corpus preprocesado para FastText**
    # #### **Preprocesamiento de texto (para TF-IDF)**
    if run_preprocess:
        logging.info("Preprocesando...")
        with Pool(POOL_SIZE) as my_pool:
            pp_list = my_pool.map(preprocessor_sentences, df.text.values)
        df["pp"] = pp_list
        if save_pp_file:
            df.to_csv(pp_file_path)
        
    # df["class_"] = pd.Categorical(df["class_"])
    # df.to_csv("data/cc_dump_full_pp_20210508.csv")
    # df = pd.read_csv("data/cc_dump_full_pp_20210508.csv")
    df[df.text.isnull()]
    df[df.pp==""]
    df = df[df.text.notnull()].copy()
    df[~np.array([isinstance(x, str) for x in df.pp.values])]
    df = df[[isinstance(x, str) for x in df.pp.values]].copy()

    doc_list = df.sentencia.unique().tolist()
    doc_sample = random.sample(doc_list, k=int(len(doc_list)*LDA_SAMPLE_SIZE))
    df_sample = df[df.sentencia.isin(doc_sample)].copy()
    logging.info(f"Tamaño de muestra: {len(df_sample)} elementos (párrafos)")

    # DESCARGAR MEMORIA!
    df = None
    # df_sample["lemma_stop"] = df_sample["pp"]
    df_sample = df_sample[[isinstance(a, str) for a in df_sample.pp.values]]
    df_sample["pp"] = [doc.replace("\n", " ").strip() for doc in df_sample.pp.values]
    # create tmp corpus file for yielding
    df_sample.pp.to_csv(LDA_CORPUS_PATH, index=False, header=False)
    
    # Build dict
    cc_dict = corpora.Dictionary(line.lower().strip().split() for line in open(LDA_CORPUS_PATH))
    logging.info("Dictionary initial size: {}".format(len(cc_dict)))

    # Filter common words in paragraphs and docs
    # paragraph level
    cc_dict.filter_extremes(no_above=0.5, keep_n=200000)
    cc_dict.compactify()  # remove gaps in id sequence after words that were removed

    # document-level
    doc_dict = {doc:' '.join(df_sample[df_sample.sentencia==doc]["pp"].values) 
                for doc in df_sample.sentencia.unique()}
    doc_word_dict = {d:list(set(doc_dict[d].strip().split())) for d in doc_dict}
    doc_df = pd.DataFrame([{"name":doc, "words":' '.join(doc_word_dict[doc])} for doc in doc_word_dict])
    cv = CountVectorizer(max_df=0.5)
    cv.fit(doc_df.words)
    logging.info(f"Document-level stopwords: {len(cv.stop_words_)}")
    stop_ids = [cc_dict.token2id[w] for w in cv.stop_words_]
    cc_dict.filter_tokens(bad_ids=stop_ids)
    cc_dict.compactify()

    # short words
    word_set = list(set([w for doc in doc_word_dict for w in doc_word_dict[doc]]))
    word_len = [(w, len(w)) for w in word_set]
    short_words = [word for word, length in word_len if length<3]
    logging.info(f"Short words to remove: {len(short_words)}")
    short_words = [w for w in short_words if w in cc_dict.token2id.keys()]
    short_ids = [cc_dict.token2id[w] for w in short_words]
    cc_dict.filter_tokens(bad_ids=short_ids)
    cc_dict.compactify()
    
    # Train
    light_corpus = CorpusIterator(LDA_CORPUS_PATH, cc_dict)
    tfidf = models.TfidfModel(light_corpus)
    corpus_tfidf = TfidfCorpus(light_corpus, tfidf)

    # Train the model on the corpus.
    lda = LdaMulticore(corpus_tfidf,
                        id2word=cc_dict,
                        num_topics=LDA_NUM_TOPICS,
                        workers=LDA_WORKERS,
                        per_word_topics=True,
                        random_state=RANDOM_STATE,
                        passes=1)
    
    
    # temp_file = datapath("model")
    cc_dict.save_as_text(LDA_DICT_PATH)
    tfidf.save(LDA_TFIDF_PATH)
    lda.save(LDA_MODEL_PATH)
