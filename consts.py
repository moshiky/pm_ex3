# Ori Rabi - 305284598
# Moshe Cohen - 203671508

import os


UNSEEN_WORD = 'lfthisnbekrnticehguifgiuhfjfkjhs'
PRECISION_LENGTH = 5
NUM_OF_CLUSTERS = 9
UNDERFLOW_VALUE = 8
EPSILON = 10 ** -5
START_LAMBDA = 10 ** -1
MIN_LAMBDA = 10 ** -12
LAMBDA_CHANGE_INTERVAL = 28
MINIMAL_INTERVAL = 10 ** -7
GRAPH_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'graphs')
LIKELIHOOD_GRAPH_FILE_NAME = 'likelihood_{iteration_id}.jpg'
