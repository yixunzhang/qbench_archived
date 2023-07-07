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

class TransformerModel(nn.Module):
    def __init__(self, d_feat=600, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None, half_precision=False):
        super(TransformerModel, self).__init__()
        self.transformer_model = torch.nn.Transformer(d_feat, nhead)
        self.device = device
        self.d_feat = d_feat
        self.half_precision = half_precision

    def forward(self, src):
        src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
        src = src.transpose(1, 0) # not batch first
        output = self.transformer_model(src[:-1],src[-1:])
        return output.transpose(1,0)[:, -1, :].squeeze()

class Transformer(Model):
    GPU_UTILS_SETTINGS = {
    "high" : {"batch_size": 16, },
    "median": {"batch_size": 8, },
    "low" : {"batch_size": 4, },}
    def __init__(
        self,
        d_model=64,
        nhead=2,
        num_layers=2,
        dropout=0,
        lr=0.0001,
        optimizer="sgd",
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
        elif optimizer.lower() == "sgd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
        if self.use_gpu:
            self.model.to(self.device)
        else:
            import intel_extension_for_pytorch as ipex
            self.model, self.train_optimizer = ipex.optimize(self.model, optimizer=self.train_optimizer, dtype=torch.bfloat16 if self.use_bf16 else torch.float32)
        if self.use_half:
            self.model.half()
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
            device_ids=[self.device.index],
            output_device=self.device.index,
            find_unused_parameters=True)

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

    def test_epoch(self, data_x, data_y):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(data_x)
            loss = self.loss_fn(pred, data_y)
            score = self.metric_fn(pred, data_y)
        return loss, score

    def fit(self):
        for _, (batch_x, batch_y) in enumerate(self.train_loader):
            if not self.measure_all and self.use_gpu:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                torch.cuda.synchronize()
            with TimeEvaluator.time_context("transformer_train_epoch", warmup=5):
                if self.measure_all and self.use_gpu:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                if self.use_half:
                    batch_x, batch_y = batch_x.half(), batch_y.half()

                if self.use_bf16:
                    with torch.cpu.amp.autocast():
                        self.train_epoch(batch_x, batch_y)
                        self.test_epoch(batch_x, batch_y)
                else:
                    self.train_epoch(batch_x, batch_y)
                    self.test_epoch(batch_x, batch_y)
                    if self.use_gpu:
                        torch.cuda.synchronize()
                    if self.distributed:
                        torch.distributed.barrier()
            if not self.check_iter():
                break
        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self):
        pass

