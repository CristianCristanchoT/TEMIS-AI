#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
from gensim.models import tfidfmodel
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore

from core.settings import *
from core.preprocess import *

# diccionarios especiales para puntuación y palabras vacias
# nltk.download('punkt') # Manejo de puntuación
# nltk.download('wordnet')
# nltk.download('stopwords')

# nlp = spacy.load("es_core_news_md")

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

class LDAEngine():
    def __init__(self):
        self.cc_dict = Dictionary.load_from_text(LDA_DICT_PATH)
        self.tfidf = tfidfmodel.TfidfModel.load(LDA_TFIDF_PATH)
        self.lda = LdaMulticore.load(LDA_MODEL_PATH)

    def get_sentence_topics(self, sentence:str):
        bow_busqueda = self.cc_dict.doc2bow(sentence.lower().strip().split())
        return self.lda.get_document_topics(bow_busqueda)

