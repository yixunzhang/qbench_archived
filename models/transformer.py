import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model import Model
from utils.profiler import TimeEvaluator

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, d_feat=600, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None, half_precision=False):
        super(TransformerModel, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model, device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat
        self.half_precision = half_precision

    def forward(self, src):
        # src [N, F*T] --> [N, T, F]
        src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
        src = self.feature_layer(src)
        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0) # not batch first
        mask = None
        src = self.pos_encoder(src)
        if self.half_precision:
            output = self.transformer_encoder(src.float(), mask).half()
        else:
            output = self.transformer_encoder(src, mask)
            # [T, N, F] --> [N, T’F]
            output = self.decoder_layer(output.transpose(1, 0)[:, -1, :]) # [512, 1]
        return output.squeeze()

class Transformer(Model):
    GPU_UTILS_SETTINGS = {
    "high" : {"batch_size": 128, },
    "median": {"batch_size": 64, },
    "low" : {"batch_size": 32, },}
    def __init__(
        self,
        d_model=64,
        nhead=2,
        num_layers=2,
        dropout=0,
        lr=0.0001,
        optimizer="adam",
        reg=1e-3,
        gpu_util=None,
        **kwargs
    ):
        super(Transformer, self).__init__(  d_model=d_model,
                                            nhead=nhead,
                                            num_layers=num_layers,
                                            dropout=dropout,
                                            lr=lr,
                                            optimizer=optimizer,
                                            reg=reg,
                                            **kwargs)
    # set hypeP-parameters.
        self.d_model = d_model
        self.reg = reg
        if gpu_util is not None:
            self.batch_size = Transformer.GPU_UTILS_SETTINGS[gpu_util]['batch_size']
        self.set_iter()
        self.logger.info("Transformer:"
                        f"\nd_feat : {self.d_feat}"
                        f"\nhidden_size : {self.hidden_size}"
                        f"\nnum_layers : {self.num_layers}"
                        f"\ndropout : {self.dropout}"
                        f"\nlr : {self.lr}"
                        f"\nbatch_size : {self.batch_size}"
                        f"\noptimizer : {self.optimizer}"
                        f"\nbatch_size : {self.batch_size}"
                        f"\ndevice : {self.device}")
        self.model = TransformerModel(self.d_feat, d_model, nhead, num_layers, dropout, self.device, self.use_half)
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
        self.model.to(self.device)
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
            device_ids=[self.device.index],
            output_device=self.device.index,
            find_unused_parameters=True)
        if self.use_half:
            self.model.half()
            self.model.transformer_encoder.float()

    def loss_fn(self, o, y):
        return torch.mean((o[..., 0] - y[..., 0]) ** 2)

    def metric_fn(self, pred, label):
        return -self.loss_fn(pred, label)

    def train_epoch(self, x_train, y_train):
        self.model.train()
        pred = self.model(x_train)
        loss = self.loss_fn(pred, y_train)
        self.train_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.6)
        self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(data_x)
            loss = self.loss_fn(pred, data_y)
            score = self.metric_fn(pred, data_y)
        return loss, score

    def fit(self):
        for _, (batch_x, batch_y) in enumerate(self.train_loader):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            if self.use_half:
                batch_x, batch_y = batch_x.half(), batch_y.half()
            with TimeEvaluator.time_context("transformer_train_epoch(no h2d copy)"):
                self.train_epoch(batch_x, batch_y)
                self.test_epoch(batch_x, batch_y)
            self.count_iter()
        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self):
        pass

