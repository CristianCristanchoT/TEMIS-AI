#!/usr/bin/env python
# coding: utf-8
import os
import sys
sys.path.append(os.getcwd())
# print(sys.path)

import logging
import os
import random
from multiprocessing import Pool
from operator import index
from string import punctuation

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py 
from gensim import corpora, models
from gensim.corpora.dictionary import Dictionary
from gensim.models import FastText
from gensim.models.ldamulticore import LdaMulticore
from gensim.test.utils import common_texts, datapath
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances

from core.preprocess import *
from core.settings import *

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def train_search(dataset_path:str = None, print_stats:bool = True, save_pp_file:bool= True, pp_file_path:str = None, run_preprocess:bool=True):
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

    if print_stats:
        print("Archivo {} cargado.".format(dataset_path))
        print("Cantidad de documentos: {}".format(len(df.sentencia.unique())))
        print("Cantidad de párrafos: {}".format(len(df)))

    df = df[[isinstance(x, str) for x in df.text.values]]

    # ## **Crear corpus preprocesado para FastText**
    # #### **Preprocesamiento de texto (para TF-IDF)**
    if run_preprocess:
        print("Preprocesando...")
        with Pool(POOL_SIZE) as my_pool:
            pp_list = my_pool.map(preprocessor_sentences, df.text.values)
        df["pp"] = pp_list
        if save_pp_file:
            df.to_csv(pp_file_path)

    for i, frase in enumerate([frase for frase in df.iloc[11111].pp.split("\n")]):
        print("{0}. {1}".format(i, frase))

    # **Guardar Corpus de documentos segmentados por frase**
    with open(CORPUS_PATH, "w") as file:
        for doc in df.pp.values:
            if isinstance(doc, str):
                file.write(doc)
                file.write("\n")
    
    carpeta = os.getcwd()
    direccion_corpus = os.path.join(carpeta, CORPUS_PATH)

    corpus_file = datapath(direccion_corpus)         # absolute path to corpus
    model = FastText(size=MODEL_DIM, 
                    window=MODEL_WINDOW, 
                    min_count=5, 
                    workers=-1, 
                    seed=RANDOM_STATE)
    model.build_vocab(corpus_file=corpus_file)      # scan over corpus to build the vocabulary
    total_words = model.corpus_total_words          # number of words in the corpus
    model.train(corpus_file=corpus_file, total_words=total_words, epochs=MODEL_EPOCHS)

    # **Guardar modelo** 
    model.save(MODEL_PATH)

if __name__=="__main__":
    train_search()
