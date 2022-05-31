import logging

logger = logging.getLogger('BertLog')
file_handler = logging.FileHandler('log.txt')
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(funcName)s(%(lineno)d): %(message)s')
logger.addHandler(file_handler)



