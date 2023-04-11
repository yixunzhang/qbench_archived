import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import intel_extension_for_pytorch as ipex

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model import Model
from utils.profiler import TimeEvaluator


class LSTMModel(nn.Module):
    def __init__(self, d_feat=600, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)
        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1) # [N, F, T]
        x = x.permute(0, 2, 1) # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()

class LSTM(Model):
    GPU_UTILS_SETTINGS = {
        "high" : {"batch_size": 64, "hidden_size": 512},
        "median": {"batch_size": 256, "hidden_size": 256},
        "low" : {"batch_size": 512, "hidden_size": 64},
    }
    def __init__(
        self,
        gpu_util=None,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        lr=0.001,
        optimizer="sgd",
        **kwargs
    ):
        super(LSTM, self).__init__(hidden_size =hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            optimizer=optimizer,
            **kwargs)

        # set hyper-parameters.
        if gpu_util is not None:
            self.batch_size, self.hidden_size = LSTM.GPU_UTILS_SETTINGS[gpu_util].values()
        self.set_iter()
        self.logger.info(
            "LSTM parameters settingz"
            f"\nd_feat : {self.d_feat}"
            f"\nhidden_size : {self.hidden_size}"
            f"\nnum_layers : {self.num_layers}"
            f"\ndropout : {self.dropout}"
            f"\nlr : {self.lr}"
            f"\nbatch_size : {self.batch_size}"
            f"\noptimizer : {self.optimizer}"
            f"\nuse_GPU : {self.use_gpu}"
            f"\nseed : {self.seed}")
        self.model = LSTMModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,)

        if self.optimizer == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(self.optimizer))
        if self.use_gpu:
            self.model.to(self.device)
        else:
            self.model, self.train_optimizer = ipex.optimize(self.model, optimizer=self.train_optimizer, dtype=torch.bfloat16 if self.use_bf16 else torch.float32)
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                            device_ids=[self.device.index],
                                                            output_device=self.device.index)
        if self.use_half:
            self.model.half()

    def loss_fn(self, o, y):
        return torch.mean((o[..., 0] - y[..., 0]) ** 2)
    
    def metric_fn(self, pred, label):
        return -self.loss_fn(pred, label)
    
    def train_epoch(self, batch_x, batch_y):
        self.model.train()
        pred = self.model(batch_x)
        loss = self.loss_fn(pred, batch_y)
        self.train_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.6)
        self.train_optimizer.step()
    
    def test_epoch(self, batch_x, batch_y):
        self.model.eval()
        pred = self.model(batch_x)
        loss = self.loss_fn(pred, batch_y)
        score = self.metric_fn(pred, batch_y)
        return loss, score

    def fit(self):
        for _, (batch_x, batch_y) in enumerate(self.train_loader):
            if self.use_gpu:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            if self.use_half:
                batch_x, batch_y = batch_x.half(), batch_y.half()
            # train
            with TimeEvaluator.time_context("lstm_train_epoch(no h2d copy)"):
                if self.use_bf16:
                    with torch.cpu.amp.autocast():
                        self.train_epoch(batch_x, batch_y)
                        self.test_epoch(batch_x, batch_y)
                else:
                    self.train_epoch(batch_x, batch_y)
                    self.test_epoch(batch_x, batch_y)
                    if self.use_gpu:
                        torch.cuda.synchronize()
            self.count_iter()
        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, input = None):
        pass
