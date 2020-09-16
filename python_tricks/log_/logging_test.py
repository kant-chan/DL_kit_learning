
import logging
import logging.config
 
logging.config.fileConfig("logging.conf")
 
def logerror():
    logger = logging.getLogger("errorLogger")
    logger.error("There is a error in this file",exc_info=1)
 
def logdebug():
    logger = logging.getLogger("debugLogger")
    logger.debug("There is a debug in this file")
 
logdebug()
 
def testfun():
    print("test")
 
try:
    testfun(1)
except TypeError as e:
    logerror()