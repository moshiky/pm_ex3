# Ori Rabi - 305284598
# Moshe Cohen - 203671508

import time
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os
from document import Document
import consts


class Utils:

    def __init__(self, output_file_path):
        self.__output_file_path = output_file_path
        self.__write_file_line("#Students Ori_Rabi Moshe_Cohen 305284598 203671508", True)
        self.__output_id = 1

    def __write_file_line(self, line, is_first_line=False):
        open_mode = 'ab'
        if is_first_line:
            open_mode = 'wb'

        with open(self.__output_file_path, open_mode) as output_file:
            output_file.write(str(line) + '\r\n')

    @staticmethod
    def log(msg):
        print "[{timestamp}] >> {msg}".format(timestamp=time.ctime(), msg=msg)

    def report(self, msg):
        line = "#Output{output_id} {msg}".format(output_id=self.__output_id, msg=msg)
        self.__write_file_line(line)
        self.__output_id += 1

        self.log(line)

    @staticmethod
    def get_filtered_text(text):
        Utils.log('get_filtered_text')
        word_counters = Counter(text.split())
        for word in word_counters.keys():
            if word_counters[word] < 4:
                text = \
                    text.replace(' ' + word + ' ', ' ').\
                    replace('\r\n' + word + ' ', '\r\n').\
                    replace(' ' + word + '\r\r\n', '\r\n')

        return text

    @staticmethod
    def load_dataset(dataset_file_path, topics_file_path):
        Utils.log('load_dataset')
        with open(dataset_file_path, 'rb') as input_file:
            raw_text = input_file.read()

        raw_text = Utils.get_filtered_text(raw_text)
        text_lines = raw_text.replace('\r\r\n', '\r\n').split('\r\n')

        # the dataset structure is ``a line with a title, and then the article itself.
        Utils.log('build documents')
        documents = list()
        line_index = 0
        while line_index+1 < len(text_lines):
            if len(text_lines[line_index]) == 0 or len(text_lines[line_index+1]) == 0\
                    or text_lines[line_index][0] != '<' or text_lines[line_index+1] == '<':
                line_index += 1

            else:
                headline = text_lines[line_index]
                document_content = text_lines[line_index+1].split(' ')
                documents.append(Document(headline, document_content))
                line_index += 2

        # load topics file
        with open(topics_file_path, 'rb') as input_file:
            raw_text = input_file.read()

        topics = filter(lambda x: len(x) > 0, raw_text.split('\r\r\n'))

        return documents, topics

    @staticmethod
    def print_likelihood_graph(likelihood_for_iteration):
        plt.clf()
        plt.plot(range(1, len(likelihood_for_iteration)+1), likelihood_for_iteration)
        plt.plot(
            [1, len(likelihood_for_iteration)+1],
            [likelihood_for_iteration[-1], likelihood_for_iteration[-1]],
            'r--'
        )
        plt.grid(True)
        plt.savefig(
            os.path.join(consts.GRAPH_FOLDER_PATH, consts.LIKELIHOOD_GRAPH_FILE_NAME).format(
                iteration_id=len(likelihood_for_iteration)
            )
        )

    @staticmethod
    def get_cluster_topic_counters(cluster, topics):
        topic_counters = dict(zip(topics, np.zeros(len(topics))))
        for document in cluster:
            document_topics = document.get_document_topics()
            for topic in document_topics:
                topic_counters[topic] += 1
        return np.array(topic_counters.values())

    @staticmethod
    def get_confusion_matrix(topics, clusters):
        """
        :param topics: list of strings. the original topics list of the supplied topics file
        :param clusters: dictionary of list of documents.
        """
        confusion_matrix = np.array(np.zeros((len(clusters), len(topics)+2)))
        cluster_topics = list()
        for cluster_index, cluster in enumerate(clusters):
            cluster_counters = Utils.get_cluster_topic_counters(cluster, topics)
            # build cluster row
            cluster_row = np.append([cluster_index], cluster_counters)
            cluster_row = np.append(cluster_row, [len(cluster)])
            # store cluster row at the matrix
            confusion_matrix[cluster_index] = cluster_row

            # get cluster info
            cluster_topic_id = Utils.get_cluster_dominant_topic_id(cluster_counters)
            cluster_topic_name = topics[cluster_topic_id]
            cluster_topic_freq = cluster_counters[cluster_topic_id]
            cluster_topics.append((cluster_index, cluster_topic_name, cluster_topic_freq))

        # sort matrix by cluster size
        confusion_matrix = confusion_matrix[confusion_matrix[:, -1].argsort()][::-1]

        return confusion_matrix, cluster_topics

    @staticmethod
    def get_cluster_dominant_topic_id(cluster_counters):
        return np.argmax(cluster_counters)
