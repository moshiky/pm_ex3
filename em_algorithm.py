# Ori Rabi - 305284598
# Moshe Cohen - 203671508

import numpy as np
import consts
from utils import Utils


class EMAlgorithm:
    def __init__(self, documents, topics):
        # self.__documents: list of Document objects
        # contains all of train documents
        self.__documents = documents

        # self.__clusters: list of strings
        # contains all of cluster names
        self.__topics = topics

        self.__clusters_probabilities = np.zeros(consts.NUM_OF_CLUSTERS)
        self.__word_for_given_cluster_probabilities = None
        self.__cluster_for_document_probabilities = None
        # self.__filter_words()
        self.__all_seen_words = self.__get_all_seen_words()
        self.__all_document_counters = \
            np.array([document.get_all_word_counters_vector(self.__all_seen_words) for document in self.__documents])
        self.__document_counter_sum = sum(self.__all_document_counters.transpose())
        Utils.log("done init instance")

    def __calculate_likelihood_log(self):
        Utils.log("__calculate_likelihood_log")

        total_likelihood = 0
        for document_index in range(len(self.__documents)):
            z_list = self.__get_zi_list(document_index)
            max_zi = max(z_list)
            document_sum = \
                sum(
                    np.exp(z_list[i] - max_zi) if z_list[i] + consts.UNDERFLOW_VALUE >= max_zi
                    else 0
                    for i in range(consts.NUM_OF_CLUSTERS)
                )
            total_likelihood += max_zi + np.log(document_sum)

        return total_likelihood

    def __filter_words(self):
        Utils.log('__filter_words')
        all_word_counters = dict()
        for document in self.__documents:
            document_word_counters = document.get_existing_word_counters()
            for word in document_word_counters.keys():
                if word not in all_word_counters.keys():
                    all_word_counters[word] = 0
                all_word_counters[word] += document_word_counters[word]

        # mark words for deletion
        words_to_delete = list()
        for word in all_word_counters:
            if all_word_counters[word] < 4:
                words_to_delete.append(word)

        # remove words from documents
        for document in self.__documents:
            for word in words_to_delete:
                document.remove_word(word)

    def __get_all_seen_words(self):
        Utils.log("__get_all_seen_words")

        all_seen_words = list()
        for document in self.__documents:
            all_seen_words += document.get_document_unique_words()

        all_seen_words = set(all_seen_words)
        Utils.log('vocabulary size= {all_seen_words_size}'.format(all_seen_words_size=len(all_seen_words)))
        return all_seen_words

    def __get_zi_list(self, document_index):
        # Utils.log("__get_zi_list")
        return np.log(self.__clusters_probabilities) + \
            self.__all_document_counters[document_index].dot(
                np.log(self.__word_for_given_cluster_probabilities.transpose())
            )

    def __execute_e_step(self):
        Utils.log("__execute_e_step")

        # for each document
        for document_index in range(len(self.__documents)):
            # for each cluster
            z_list = self.__get_zi_list(document_index)
            max_zi = max(z_list)

            for i in range(consts.NUM_OF_CLUSTERS):
                # calculate cluster probability given the document
                if z_list[i] + consts.UNDERFLOW_VALUE < max_zi:
                    self.__cluster_for_document_probabilities[document_index][i] = 0
                else:   # z_list[i] + consts.UNDERFLOW_VALUE >= max_zi
                    numerator = np.exp(z_list[i] - max_zi)
                    denominator = \
                        sum([
                                np.exp(z_list[j] - max_zi) if z_list[j] + consts.UNDERFLOW_VALUE >= max_zi
                                else 0
                                for j in range(consts.NUM_OF_CLUSTERS)
                        ])
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
        Utils.log("alpha sum= {alpha_sum}".format(alpha_sum=sum(self.__clusters_probabilities)))
        self.__clusters_probabilities /= sum(self.__clusters_probabilities)
        Utils.log("normalized alpha sum= {alpha_sum}".format(alpha_sum=sum(self.__clusters_probabilities)))

        # calculate Pik - probability of each word for each cluster
        Utils.log("calculate Pik")

        Utils.log("calculate numerator matrix")
        numerator_matrix = \
            self.__cluster_for_document_probabilities.transpose().dot(self.__all_document_counters) + \
            consts.LAMBDA

        Utils.log("calculate denominator vector")
        denominator_vector = \
            self.__document_counter_sum.dot(self.__cluster_for_document_probabilities) + \
            len(self.__all_seen_words) * consts.LAMBDA \

        Utils.log("calculate word for cluster probabilities matrix")
        self.__word_for_given_cluster_probabilities = (numerator_matrix.transpose() / denominator_vector).transpose()

    def __get_clusters(self):
        clusters = [[] for i in range(consts.NUM_OF_CLUSTERS)]
        for document_index, document in enumerate(self.__documents):
            clusters[self.__get_document_most_likely_cluster_id(document_index)].append(document)

        return clusters

    def __get_loss(self, clusters):
        mistakes = 0.0
        for cluster_id, cluster in enumerate(clusters):
            cluster_counters = Utils.get_cluster_topic_counters(cluster, self.__topics)
            cluster_topic_id = Utils.get_cluster_dominant_topic_id(cluster_counters)
            cluster_dominant_topic = self.__topics[cluster_topic_id]
            dominant_topic_freq = cluster_counters[cluster_topic_id]
            Utils.log('cluster #{cluster_id} dominant topic: {topic}, freq={freq}'.format(
                    topic=cluster_dominant_topic, cluster_id=cluster_id, freq=dominant_topic_freq
                )
            )
            mistakes += len(cluster) - dominant_topic_freq

        return mistakes / len(self.__documents)

    def __get_document_most_likely_cluster_id(self, document_index):
        return np.argmax(self.__cluster_for_document_probabilities[document_index])

    def run(self):
        Utils.log("run")

        # initiate clustering
        self.__cluster_for_document_probabilities = np.zeros(shape=(len(self.__documents), consts.NUM_OF_CLUSTERS))
        for i in range(consts.NUM_OF_CLUSTERS):
            self.__cluster_for_document_probabilities[i::consts.NUM_OF_CLUSTERS, i] = 1

        # M step
        self.__execute_m_step()

        # iterate until progress is too small
        likelihood_history = list()
        likelihood_history.append(self.__calculate_likelihood_log() - (consts.MINIMAL_INTERVAL + 1))
        likelihood_history.append(likelihood_history[0] + (consts.MINIMAL_INTERVAL + 1))
        Utils.log('first likelihood = {last_likelihood}'.format(last_likelihood=likelihood_history[-1]))

        while likelihood_history[-1] - likelihood_history[-2] > consts.MINIMAL_INTERVAL:

            # E step
            self.__execute_e_step()

            # M step
            self.__execute_m_step()

            # calculate likelihood log
            likelihood_history.append(self.__calculate_likelihood_log())

            # build clusters
            clusters = self.__get_clusters()

            # print current performance
            Utils.log('iteration #{iteration_id}: new_likelihood= {new_likelihood}, diff={diff}, acc={acc}'.format(
                    iteration_id=len(likelihood_history)-1, new_likelihood=likelihood_history[-1],
                    diff=likelihood_history[-1]-likelihood_history[-2], acc=1-self.__get_loss(clusters)
                )
            )

            # save graph
            Utils.print_likelihood_graph(likelihood_history[1:])

            # print confusion matrix
            Utils.log('\n' + str(Utils.get_confusion_matrix(self.__topics, clusters)))
