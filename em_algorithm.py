
import numpy as np
import consts
from utils import Utils


class EMAlgorithm:
    def __init__(self, documents, clusters):
        self.__documents = documents
        self.__clusters = clusters
        self.__clusters_probabilities = np.zeros(consts.NUM_OF_CLUSTERS)
        self.__word_for_given_cluster_probabilities = None
        self.__cluster_for_document_probabilities = None
        self.__all_seen_words = self.__get_all_seen_words()
        Utils.log("here")
        self.__all_document_counters = \
            np.array([document.get_all_word_counters_vector(self.__all_seen_words) for document in self.__documents])
        Utils.log("here2")
        self.__document_counter_sum = sum(self.__all_document_counters.transpose())
        Utils.log("here3")

    def __calculate_likelihood_log(self):
        Utils.log("__calculate_likelihood_log")

        total_likelihood = 0
        for document_index in range(len(self.__documents)):

            z_list = self.__get_zi_list(document_index)
            max_zi = max(z_list)

            document_likelihood = 0
            for i in range(consts.NUM_OF_CLUSTERS):
                document_likelihood += np.exp(z_list[i] - max_zi)

            total_likelihood += np.log(document_likelihood)

        return total_likelihood

    def __get_all_seen_words(self):
        Utils.log("__get_all_seen_words")

        all_seen_words = list()
        for document in self.__documents:
            all_seen_words += document.get_document_unique_words()

        return set(all_seen_words)

    def __get_zi_list(self, document_index):
        # Utils.log("__get_zi_list")
        return np.log(self.__clusters_probabilities) + \
            self.__all_document_counters[document_index].dot(
                np.log(self.__word_for_given_cluster_probabilities.transpose())
            )

        # document_word_counters = document.get_word_counters()
        #
        # # <<<<<<<<
        # for word in document_word_counters.keys():
        #     total += document_word_counters[word] * np.log(self.__word_for_given_cluster_probabilities[i][word])

    def __execute_e_step(self):
        Utils.log("__execute_e_step")

        # for each document
        for document_index in range(len(self.__documents)):
            # for each cluster
            z_list = self.__get_zi_list(document_index)
            max_zi = max(z_list)

            denominator = sum([np.exp(z_list[j] - max_zi) for j in range(consts.NUM_OF_CLUSTERS)])
            for i in range(consts.NUM_OF_CLUSTERS):
                # calculate cluster probability given the document
                if z_list[i] - max_zi < -consts.UNDERFLOW_VALUE:
                    self.__cluster_for_document_probabilities[document_index][i] = 0
                else:
                    numerator = np.exp(z_list[i] - max_zi)
                    self.__cluster_for_document_probabilities[document_index][i] = numerator / denominator

    @staticmethod
    def __add_epsilon_to_zeros(x):
        return x if x > 0 else consts.EPSILON

    def __execute_m_step(self):
        Utils.log("__execute_m_step")

        # calculate alpha i - average probability of each cluster
        Utils.log("calculate alpha")
        self.__clusters_probabilities = sum(self.__cluster_for_document_probabilities) / len(self.__documents)
        vectorized_func = np.vectorize(EMAlgorithm.__add_epsilon_to_zeros)
        self.__clusters_probabilities = vectorized_func(self.__clusters_probabilities)
        self.__clusters_probabilities /= sum(self.__clusters_probabilities)

        # calculate Pik - probability of each word for each cluster
        Utils.log("calculate Pik")

        Utils.log("calculate up total")
        up_total_matrix = \
            self.__cluster_for_document_probabilities.transpose().dot(self.__all_document_counters) + \
            consts.LAMBDA

        Utils.log("calculate down total")
        down_total_vector = \
            self.__document_counter_sum.dot(self.__cluster_for_document_probabilities) + \
            consts.VOCABULARY_SET_LENGTH * consts.LAMBDA

        Utils.log("calculate word for cluster probabilities matrix")
        self.__word_for_given_cluster_probabilities = (up_total_matrix.transpose() / down_total_vector).transpose()

        # self.__word_for_given_cluster_probabilities = [dict()]*consts.NUM_OF_CLUSTERS
        #
        # for cluster_index in range(consts.NUM_OF_CLUSTERS):
        #     Utils.log("calculate for cluster " + str(cluster_index))
        #     self.__word_for_given_cluster_probabilities[cluster_index] = dict()
        #
        #     Utils.log("calculate down total")
        #     cluster_for_document_probabilities = self.__cluster_for_document_probabilities[:, cluster_index]
        #     down_total = consts.VOCABULARY_SET_LENGTH * consts.LAMBDA
        #     down_total += \
        #         cluster_for_document_probabilities.dot(
        #             np.array([sum(document_counters) for document_counters in self.__all_document_counters])
        #         )
        #
        #     Utils.log("calculate for each word: " + str(len(self.__all_seen_words)))
        #     up_total = consts.LAMBDA
        #     cluster_for_document_probabilities = self.__cluster_for_document_probabilities[:, cluster_index]
        #
        #     for word in self.__all_seen_words:
        #         up_total = consts.LAMBDA
        #         for document in self.__documents:
        #             up_total += \
        #                 self.__cluster_for_document_probabilities[document.get_document_id()][i] * \
        #                 (document.get_existing_word_counters()[word] if word in document.get_document_unique_words()
        #                  else 0)
        #
        #         self.__word_for_given_cluster_probabilities[i][word] = up_total/down_total

    def run(self):
        Utils.log("run")

        # initiate clustering
        self.__cluster_for_document_probabilities = np.zeros(shape=(len(self.__documents), consts.NUM_OF_CLUSTERS))
        for i in range(consts.NUM_OF_CLUSTERS):
            self.__cluster_for_document_probabilities[i::consts.NUM_OF_CLUSTERS, i] = 1

        # M step
        self.__execute_m_step()

        # iterate until progress is too small
        last_likelihood = self.__calculate_likelihood_log()
        Utils.log('first likelihood = {last_likelihood}'.format(last_likelihood=last_likelihood))
        new_likelihood = last_likelihood - (consts.MINIMAL_INTERVAL + 1)
        while last_likelihood - new_likelihood > consts.MINIMAL_INTERVAL:
            # E step
            self.__execute_e_step()

            # M step
            self.__execute_m_step()

            last_likelihood = new_likelihood
            new_likelihood = self.__calculate_likelihood_log()
            Utils.log('new_likelihood= {new_likelihood}, diff={diff}'.format(
                new_likelihood=new_likelihood, diff=last_likelihood-new_likelihood)
            )
