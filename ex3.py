
# Ori Rabi - 305284598
# Moshe Cohen - 203671508

import os
import sys
from utils import Utils


def main(development_dataset_file_path, topics_file_path):
    documents, topics = Utils.load_dataset(development_dataset_file_path, topics_file_path)
    print len(documents)
    print len(topics)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        Utils.log('usage: {file_name} <development_dataset_file_path> <topics_file_path>'.format(
            file_name=os.path.basename(__file__))
        )
    else:
        main(sys.argv[1], sys.argv[2])