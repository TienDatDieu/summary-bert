import logging
import sys

logger = logging.getLogger('BertLog')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('log.txt')
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(funcName)s(%(lineno)d): %(message)s')
logger.addHandler(file_handler)
logger.addHandler(stream_handler)



