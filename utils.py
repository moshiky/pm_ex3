
# Ori Rabi - 305284598
# Moshe Cohen - 203671508

import time
from collections import Counter
from document import Document


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
                text = text.replace(' ' + word + ' ', ' ')
                text = text.replace('\r\n' + word + ' ', '\r\n')
                text = text.replace(' ' + word + '\r\n', '\r\n')

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

