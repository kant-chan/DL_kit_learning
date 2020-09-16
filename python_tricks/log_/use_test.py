import logging


"""
Logger(1) --> Handler(n)
          --> Filter(n)

           Handler(1) --> Filter(1)
           Handler(1) --> Formatter(1)
"""
# logging.basicConfig(filename='logger.log', level=logging.INFO)

logger = logging.getLogger('use_test')

fh = logging.FileHandler('/log/use_test.log')
fh.setLevel(logging.WARN)

fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)

logging.debug('debug message')
logging.info('info message')
logging.warn('warn message')
logging.error('error message')
logging.critical('critical message')
