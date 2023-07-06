import logging
class LoggerManager:
    def __init__(self, module_name, level, logpath, enable_print, rank):
        self.module_name = module_name
        self.level = level
        self.logpath = logpath
        self.rank = rank
        self.enable_print = enable_print

    @property
    def logger(self):
        logger = logging.getLogger(self.module_name)
        logger.setLevel(self.level)
        if self.rank >= 0:
            formatter = logging.Formatter(f"[%(asctime)s][rank-{self.rank}] %(message)s")
        else:
            formatter = logging.Formatter("[%(asctime)s] %(message)s")
        if self.enable_print:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
        if self.logpath is not None:
            file_handler = logging.FileHandler(self.logpath)
            file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

class _LoggerManager:
    logger = None
    def __call__(*args, **kwargs):
        if _LoggerManager.logger is None:
            _LoggerManager.logger = LoggerManager(__name__, logging.INFO, None, True).logger
        return _LoggerManager.logger

    def init(module_name, level = None, logpath = None, enable_print=False, rank=-1):
        _LoggerManager.logger = LoggerManager(module_name, level, logpath, enable_print, rank).logger

get_logger = _LoggerManager()

def init_log(module_name, level = None, logpath = None, enable_print=False, rank=-1):
    _LoggerManager.init(module_name, level, logpath, enable_print, rank)
