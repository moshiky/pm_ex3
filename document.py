
# Ori Rabi - 305284598
# Moshe Cohen - 203671508

import numpy as np
from collections import Counter, OrderedDict


class Document:

    def __init__(self, headline, content):
        self.__init_document_metadata(headline)
        self.__word_counters = Document.get_word_counters(content)
        self.__filter_content()

    def __init_document_metadata(self, headline):
        headline_parts = headline.replace('>', '').split('\t')
        self.__id = int(headline_parts[1])
        self.__topics = headline_parts[2:]

    def __filter_content(self):
        self.__word_counters = OrderedDict(sorted({
            key: value for key, value in self.__word_counters.items()
            if key not in ['', ' '] and value > 3
        }.items()))

    def get_existing_word_counters(self):
        return self.__word_counters

    def get_all_word_counters_vector(self, all_seen_words):
        return np.array([self.__word_counters[word] if word in self.__word_counters.keys() else 0
                         for word in all_seen_words])

    def get_document_unique_words(self):
        return self.__word_counters.keys()

    def get_document_id(self):
        return self.__id

    def get_document_topics(self):
        return self.__topics

    @staticmethod
    def get_word_counters(dataset):
        return OrderedDict(Counter(dataset))
