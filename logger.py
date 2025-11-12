# -*- coding: utf-8 -*-
r"""
    logger.py
    Adriano Poci
    University of Oxford
    2025

    Synopsis
    --------
    Provides a harmonized, central logging interface for all CubeFit modules.
    Can be used as a context manager to capture all output in a logfile.

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>
"""

import sys
import threading
import datetime
from pathlib import Path
import traceback
from contextlib import contextmanager, redirect_stdout, redirect_stderr

class CubeFitLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, logfile=None, mode="a"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.init(logfile, mode)
            else:
                if logfile is not None and cls._instance.logfile != Path(logfile):
                    print(f"[CubeFitLogger WARNING] Attempt to reinitialize "
                          f"logger with {logfile}, but already initialized "
                          f"with {cls._instance.logfile}. Ignoring new logfile!", file=sys.__stdout__)
            return cls._instance

    def init(self, logfile=None, mode="a"):
        if logfile is not None:
            logfile = Path(logfile)
            logfile.parent.mkdir(parents=True, exist_ok=True)
            if mode == "w":
                logfile.write_text("")  # clear the file
        self.logfile = Path(logfile) if logfile is not None else None
        self.mode = mode

    def log(self, msg="", flush=True, ts=True):
        # accept any object (including Exceptions) and stringify safely
        try:
            text = str(msg)
        except Exception:
            text = repr(msg)
        now = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]") if ts else ""
        outmsg = (now + " " + text) if ts else text
        with threading.Lock():
            print(outmsg, file=sys.__stdout__, flush=flush)
            if self.logfile is not None:
                with self.logfile.open("a") as f:
                    f.write(outmsg + "\n")
            if flush:
                sys.__stdout__.flush()

    def log_exc(self, exc: BaseException, prefix="[EXC]"):
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.log(f"{prefix} {tb}", flush=True)

    def flush(self):
        sys.__stdout__.flush()
        if self.logfile is not None:
            with self.logfile.open("a"):
                pass

    @contextmanager
    def capture_all_output(self):
        original_excepthook = sys.excepthook

        def log_excepthook(exc_type, exc_value, exc_traceback):
            exception_msg = ''.join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )
            self.log("[Exception Captured]\n" + exception_msg, flush=True)
            # Optionally call original excepthook too
            original_excepthook(exc_type, exc_value, exc_traceback)

        sys.excepthook = log_excepthook

        class LoggerStream:
            def __init__(self, logger):
                self.logger = logger

            def write(self, message):
                message = message.rstrip()
                if message:
                    self.logger.log(message, flush=True, ts=True)

            def flush(self):
                pass  # Implement if needed

        try:
            with redirect_stdout(LoggerStream(self)), redirect_stderr(LoggerStream(self)):
                yield
        finally:
            sys.excepthook = original_excepthook

# Singleton access
_logger_instance = None
_logger_lock = threading.Lock()

def get_logger(logfile=None, mode="a"):
    global _logger_instance
    with _logger_lock:
        if _logger_instance is None:
            _logger_instance = CubeFitLogger(logfile, mode)
        return _logger_instance
