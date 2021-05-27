#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import random
import re
import h5py
import gensim

from gensim.models import FastText
from gensim.models.fasttext import FastTextKeyedVectors
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances

from core.settings import *
from core.preprocess import *

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def get_paragraphs(dataframe, document, paragraph, window):
    doc_p_list = np.sort(dataframe[dataframe["sentencia"]==document]["p_index"].values)
    paragraph_i = np.where(doc_p_list == paragraph)[0][0]
    p_indices = doc_p_list[paragraph_i-window//2 : paragraph_i+window//2 + 1]
    range_from, range_to = min(p_indices), max(p_indices)
    paragraphs = dataframe.query(f'sentencia==@document & p_index>=@range_from & p_index<=@range_to')["text"]
    paragraphs = [re.sub("\s+", " ", p) for p in paragraphs]
    return '\n'.join(paragraphs)

def test_sentence_preprocessor(df, id_ejemplo:int = None):
    if id_ejemplo is None:
        id_ejemplo = 1111
    print("Original:\n", df.iloc[id_ejemplo].text)
    print(preprocessor_sentences(df.iloc[id_ejemplo].text, debug=True))


def wv_vectorizer(wv: FastTextKeyedVectors, text, preprocess:bool=False):
    """
    Calcula el vector promedio de las palabras de la frase.
    """
    if not isinstance(text, str):
        return np.zeros(wv.vector_size)

    if preprocess:
        text = full_preprocess(text)
    
    vec_list = []
    word_list = [w for w in text.split() if (w not in ES_STOPWORDS) and (w in wv.vocab)]
    for word in word_list:
        # print("{0}:\n{1}".format(word, wv[word]))
        vec_list.append(wv[word])

    if len(vec_list)==0:
        return np.random.random(wv.vector_size)

    vec_list = np.array(vec_list)
    norms = np.linalg.norm(vec_list, axis=1).reshape(-1,1)
    norms = np.array([n if n>0 else 1.0 for n in norms])
    vec_over_norm = vec_list/norms
    vec = np.mean(vec_over_norm, axis=0)
    return vec


class VectorizerWrapper():
    def __init__(self, wv):
        self.wv = wv
    
    def __call__(self, text):
        return wv_vectorizer(self.wv, text)


class SearchEngine():
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        # self.model = FastText.load(model_path)
        self.wv = gensim.models.fasttext.FastTextKeyedVectors.load(WV_PATH)
        self.df = pd.DataFrame()
        self.vectors = np.zeros((2,2))
        self.indices = pd.DataFrame()
        self.vectorizer = VectorizerWrapper(self.wv)

    def calculate_doc_vectors(self, df: pd.DataFrame, column:str = "pp"):
        with Pool(5) as poo:
            vectors = poo.map(self.vectorizer, df[column].values)

        self.vectors = np.array([v for v in vectors])
        self.indices = df[["sentencia", "p_index"]].copy()
    
    def save_vectors(self, vectors_path:str=VECTORS_PATH, index_path:str=VECTORS_INDEX_PATH):
        # with h5py.File(vectors_path, 'w') as hf_object:
        #     hf_object.create_dataset('vectors', data=self.vectors)
        pd.DataFrame(self.vectors).to_hdf(vectors_path, 'vectors', mode='w', format='table')
        self.indices.to_hdf(index_path, 'indexes', mode='w', format='table', data_columns=True)
        # hf_object.create_dataset('indexes', data=self.indices)
        # np.savetxt(vectors_path, self.vectors)
        # self.indices.to_csv(index_path, header=True, index=False)

    
    def load_vectors(self, vectors_path:str=VECTORS_PATH, index_path:str=VECTORS_INDEX_PATH):
        # with h5py.File(vectors_path, 'r') as hf_object:
        #     self.vectors = hf_object.get('vectors')
        # print(type(self.vectors))
        self.vectors = pd.read_hdf(vectors_path, 'vectors', mode='r').to_numpy()
        self.indices = pd.read_hdf(index_path, 'indexes', mode='r')
        print(self.vectors.shape)
        print(len(self.indices))

    
    def calculate_sentence_vectors(self, sentence):
        return wv_vectorizer(self.wv, sentence)


    def search(self, sentence, df):
        # frase = "pensión de sobreviviente decreto 2728 de 1969"
        # frase = "falsos positivos"
        # frase = "protección al adulto mayor"
        # frase = "pensión de supervivencia"
        # frase = "consulta previa"


        # calcular el vector de la frase
        sentence_vec = self.calculate_sentence_vectors(sentence)
        # print(sentence_vec.shape)
        # print(sentence_vec)
        # Calcular distancia de la frase a los documentos
        print("Calculando distancias...")
        distances = cosine_distances(sentence_vec.reshape(1, -1), self.vectors)
        # print(distances.shape)
        # print(type(distances))
        distances = distances.flatten()
        # print(distances.shape)

        top_docs_full = np.argsort(distances, )
        # print(top_docs_full[:10])
        sorted_distances = np.sort(distances)
        # print(sorted_distances[:10])

        # docs_over_t = np.sum(sorted_distances<(1-THRESHOLD_SIM))
        # n_top_docs = len(self.indices.iloc[top_docs_full[:docs_over_t]].sentencia.unique())
        n_top_docs = N_TOP_DOCS

        print(f"Retornando {n_top_docs} documentos.")

        size = n_top_docs
        while len(self.indices.iloc[top_docs_full[:size]].sentencia.unique()) < n_top_docs:
            size += 1
        top_docs = top_docs_full[:size]
        top_dist = sorted_distances[:size]

        # top_sentencias = self.indices.iloc[top_docs].sentencia.unique()
        top_p_index = self.indices.iloc[top_docs].sentencia.unique()

        cache_df = df[df.sentencia.isin(top_p_index)].copy()

        top_df = self.indices.iloc[top_docs].copy()
        top_df["distance"] = top_dist
        top_df.drop_duplicates(subset="sentencia", keep="first", inplace=True, ignore_index=True)

        for i, doc in enumerate(zip(top_df.sentencia, top_df.p_index, top_df.distance)):
            print(f"{i+1}. {doc[0]} - párrafo {doc[1]} (Similaridad {1-doc[2]: .1%}):\n")
            print(get_paragraphs(cache_df, doc[0], doc[1], P_WINDOW_SIZE))
            print("-"*50)

if __name__=="__main":
    se = SearchEngine()
    se.calculate_doc_vectors()
    se.save_vectors()
