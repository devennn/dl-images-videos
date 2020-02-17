import logging
import os
from pathlib import Path

def create_log_file(format, name):
    path = str(Path('.').parent.absolute())
    filename = '{}.log'.format(name)
    path = os.path.join(path, 'logs', filename)

    # define file handler and set formatter
    formatter = logging.Formatter(format)
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)

    return file_handler

def set_logger(name, level='Warnings'):
    FORMAT = '%(asctime)s : %(levelname)s : %(name)s : %(message)s'
    logging.basicConfig(format=FORMAT)

    # Gets or creates a logger
    logger = logging.getLogger(name)

    # set log level
    if(level == 'Debug'):
        logger.setLevel(level=logging.DEBUG)
    elif(level == 'Info'):
        logger.setLevel(level=logging.INFO)
    elif(level == 'Warning'):
        logger.setLevel(level=logging.WARNING)
    elif(level == 'Error'):
        logger.setLevel(level=logging.ERROR)
    elif(level == 'Critical'):
        logger.setLevel(level=logging.CRITICAL)

    # Create file handler and add to handler
    file_handler = create_log_file(format=FORMAT, name=name)
    logger.addHandler(file_handler)

    return logger
