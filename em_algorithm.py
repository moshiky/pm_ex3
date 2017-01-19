
import numpy as np
import consts


class EMAlgorithm:
    def __init__(self, documents, clusters):
        self.__documents = documents
        self.__clusters = clusters
        self.__clusters_probabilities = np.zeros(consts.NUM_OF_CLUSTERS)
        self.__word_for_given_cluster_probabilities = list()
        self.__cluster_for_document_probabilities = dict()
        self.__all_seen_words = self.__get_all_seen_words()

    def __get_all_seen_words(self):
        all_seen_words = list()
        for document in self.__documents:
            all_seen_words += document.get_document_unique_words()

        return set(all_seen_words)

    def __get_zi(self, i, document):
        total = np.log(self.__clusters_probabilities[i])
        document_word_counters = document.get_word_counters()

        for word in document_word_counters.keys():
            total += document_word_counters[word] * np.log(self.__word_for_given_cluster_probabilities[i][word])

        return total

    def __execute_e_step(self):
        # for each document
        for document in self.__documents:
            # for each cluster
            z_list = [self.__get_zi(i, document) for i in range(consts.NUM_OF_CLUSTERS)]
            max_zi = max(z_list)

            denominator = sum([np.exp(z_list[j] - max_zi) for j in range(consts.NUM_OF_CLUSTERS)])
            for i in range(consts.NUM_OF_CLUSTERS):
                # calculate cluster probability given the document
                if z_list[i] - max_zi < -consts.UNDERFLOW_VALUE:
                    self.__cluster_for_document_probabilities[document.get_document_id()][i] = 0
                else:
                    numerator = np.exp(z_list[i] - max_zi)
                    self.__cluster_for_document_probabilities[document.get_document_id()][i] = numerator / denominator

    def __execute_m_step(self):
        # alpha i - average probability of each cluster
        self.__clusters_probabilities = sum(self.__cluster_for_document_probabilities.values()) / len(self.__documents)

        # probability of each word for each cluster
        self.__word_for_given_cluster_probabilities = [dict()]*consts.NUM_OF_CLUSTERS
        for i in range(consts.NUM_OF_CLUSTERS):
            self.__word_for_given_cluster_probabilities[i] = dict()

            down_total = 0
            for document in self.__documents:
                down_total += \
                    self.__cluster_for_document_probabilities[document.get_document_id()][i] * \
                    sum(document.get_word_counters().values())

            for word in self.__all_seen_words:
                up_total = 0
                for document in self.__documents:
                    up_total += \
                        self.__cluster_for_document_probabilities[document.get_document_id()][i] * \
                        document.get_word_counters()[word]

                self.__word_for_given_cluster_probabilities[i][word] = up_total/down_total

    def run(self):
        # initiate clustering
        for i in range(consts.NUM_OF_CLUSTERS):
            cluster_documents = self.__documents[i::consts.NUM_OF_CLUSTERS]
            for document in cluster_documents:
                self.__cluster_for_document_probabilities[document.get_document_id()] = np.zeros(consts.NUM_OF_CLUSTERS)
                self.__cluster_for_document_probabilities[document.get_document_id()][i] = 1

        # M step
        self.__execute_m_step()

        # E step
        self.__execute_e_step()

