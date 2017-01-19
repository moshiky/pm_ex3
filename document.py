
# Ori Rabi - 305284598
# Moshe Cohen - 203671508

from collections import Counter, OrderedDict


class Document:

    def __init__(self, headline, content):
        self.__init_document_metadata(headline)
        self.__word_counters = Document.get_word_counters(content)
        self.__filter_content()

    def __init_document_metadata(self, headline):
        headline_parts = headline.split('\t')
        self.__id = int(headline_parts[1])
        self.__topics = headline_parts[2:]

    def __filter_content(self):
        self.__word_counters = OrderedDict(sorted({
            key: value for key, value in self.__word_counters.items()
            if key not in ['', ' '] and value > 3
        }.items()))

    @staticmethod
    def get_word_counters(dataset):
        return OrderedDict(Counter(dataset))
