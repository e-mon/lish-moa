import os
import numpy as np
import random
import torch
import logging
import logging.handlers
from contextlib import contextmanager
import time
import git
from pathlib import Path


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class LoggerFactory(metaclass=Singleton):
    def __init__(self, log_path: str = None, loglevel=logging.INFO):
        self.loglevel = loglevel
        if log_path is None:
            self.log_path = Path('./log')
        else:
            self.log_path = Path(log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def getLogger(self, log_name):
        fmt = '%(asctime)s [%(name)s|%(levelname)s] %(message)s'
        formatter = logging.Formatter(fmt)
        logger = logging.getLogger(log_name)

        # add stream Handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # add file Handler
        handler = logging.handlers.RotatingFileHandler(filename=self.log_path, maxBytes=2 * 1024 * 1024 * 1024, backupCount=10)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.setLevel(self.loglevel)

        return logger


@contextmanager
def timer(name, logger):
    t0 = time.time()
    logger.debug(f'[{name}] start')
    yield
    logger.debug(f'[{name}] done in {time.time() - t0:.0f} s')


def get_current_commit_hash():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha
