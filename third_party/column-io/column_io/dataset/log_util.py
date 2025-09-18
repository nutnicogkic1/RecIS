# -*- coding: utf-8 -*-
import os,sys,logging
import logging.handlers

class LogLevel: # Align with { DEBUG, INFO, WARNING, ERROR, FATAL }; in odps/plugins/logging.h
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    FATAL = 4
    @staticmethod
    def to_level_name(level):
        # type: (int) -> str
        return {
            LogLevel.DEBUG: "DEBUG",
            LogLevel.INFO: "INFO",
            LogLevel.WARNING: "WARNING",
            LogLevel.ERROR: "ERROR",
            LogLevel.FATAL: "FATAL",
        }[level]


def init_openstorage_logger():
    global logger, varlogger, LOG_DIR, _logger_configured
    if _logger_configured:
        return
    if not os.environ.get("STD_LOG_DIR") or os.environ.get("STD_LOG_DIR") == "":
        os.environ["STD_LOG_DIR"] = os.path.join(os.getcwd(), "log/")
        if not os.path.exists(os.environ["STD_LOG_DIR"]):
            os.makedirs(os.environ["STD_LOG_DIR"], exist_ok=True)
    LOG_DIR = os.environ["STD_LOG_DIR"]

    log_level_list = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.FATAL]
    log_level = int(os.environ.get("NEBULA_IO_LOG_LEVEL", 1)) # Default 1==INFO
    log_level = max(LogLevel.DEBUG, min(LogLevel.FATAL, log_level))
    # ---- 1. init logger ---- #
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(LogLevel.to_level_name(log_level_list[log_level]))

    log_handlers = os.environ.get("NEBULA_IO_LOG_HANDLER", "STREAM") # STREAM FILE or STREAM,FILE
    log_handlers = log_handlers.split(",")
    logfile_path = os.environ.get("STD_LOG_DIR")
    logfile_name = os.environ.get("STD_LOG_NAME", "{}.log".format(__name__))
    if "STREAM" in log_handlers or \
        not any(["STREAM" in log_handlers, "FILE" in log_handlers]) :
        ch = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if "FILE" in log_handlers:
        fh = logging.FileHandler(os.path.join(logfile_path, logfile_name))
        # E.g. [2025-01-02 12:34:56.996] [INFO] [13#443] [column_io/dataset/log.py:33] hello world
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(thread)d#%(process)d] [%(filename)s:%(lineno)d] %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    _logger_configured = True
    logger.info("logger:{} init finish".format(__name__))

    # ---- 2. init varlogger ---- #
    varlogger = logging.getLogger(__name__ + ".varlogger")
    varlogger.propagate = False
    varlogger.setLevel(LogLevel.to_level_name(log_level_list[log_level]))
    logfile_path = os.path.join(os.environ.get("STD_LOG_DIR"), "nebulaio_logs/") # os.path.dirname("/var/log/nebulaio_logs/")
    logfile_name = "io-p{}.log".format(os.getpid())
    if not os.path.exists(logfile_path):
        try:
            os.makedirs(logfile_path)
        except OSError as e:
            pass
    log_handler = logging.handlers.RotatingFileHandler(os.path.join(logfile_path, logfile_name), maxBytes=1024 * 1024 * 1024, backupCount=2) # 1G * 2 * 16 card/ machine, 32G is enough
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(thread)d#%(process)d] [%(filename)s:%(lineno)d] %(message)s")
    log_handler.setFormatter(formatter)
    varlogger.addHandler(log_handler)
    varlogger.info("varlogger:{} init finish".format(varlogger.name))

    # ---- 99. all done ---- #

_logger_configured = False # to activate multiprocess init
logger = None # type: logging.Logger
varlogger = None # type: logging.Logger # this logger will print into /var/log/ for analysis when taks finish
LOG_DIR = None # type: str # this var is for other file, not for this log.py
init_openstorage_logger()
