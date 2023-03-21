from math import ceil
import re
from utils.data import TrainLoader
from utils.log import get_logger
import numpy as np
import torch

class Model:
    def __init__(
        self,
        d_feat=600,
        hidden_size=64,
        num_layers=2,
        dropout=9.0,
        lr=0.001,
        batch_size=2000,
        optimizer="adam",
        device="cpu",
        seed=None,
        precision="fp32",
        data_size = 5 * 10 ** 6,
        select_size = 5 * 16 ** 6,
        **kwargs
    ):
        # set hypeP-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = optimizer.lower()
        self.seed = seed
        self.precision = precision
        self.data_size = data_size
        self.select_size = select_size
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        self.logger = get_logger()
        if re.match("cpu", device) or re.match("cuda:\d+$", device):
            self.device = torch.device(device)
        else:
            devices = device.split(",")
            self.local_rank = torch.distributed.get_rank()
            self.device = torch.device(devices[self.local_rank%(len(devices))])

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    @property
    def use_half(self):
        return self.precision.lower() == "fp16"

    @property
    def distributed(self):
        return hasattr(self, 'local_rank')
    
    def set_iter(self):
        if self.distributed:
            self.iter_total = ceil(self.select_size / self.batch_size / torch.distributed.get_world_size())
        else:
            self.iter_total = ceil(self.select_size / self.batch_size)
        self.iter_current = 0
        self.pct_interval = 5
        self.pct_next = self.pct_interval

    def count_iter(self):
        self.iter_current += 1
        if self.iter_current / self.iter_total * 100 >= self.pct_next:
            self.logger.info(f"Train: {self.pct_next}%")
            self.pct_next += self.pct_interval

    def train_loader_init(self, data_set, num_workers, local):
        self.train_loader = TrainLoader(self.batch_size, self.data_size, self.select_size, self.distributed)(data_set, num_workers, local)

    def fit(self):
        raise NotImplementedError()

    def predict(self, input=None):
        raise NotImplementedError()
