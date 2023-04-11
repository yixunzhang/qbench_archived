import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model import Model
from utils.profiler import TimeEvaluator

class ALSTM(Model):
    GPU_UTILS_SETTINGS = {
    "high" : {"batch_size": 256, "hidden_size": 128},
    "median": {"batch_size": 512, "hidden_size": 64},
    "low" : {"batch_size": 1024, "hidden_size": 32},
    }
    def __init__(
        self,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        lr=0.001,
        batch_size=2000,
        optimizer="adam",
        gpu_util=None,
        **kwargs    
    ):
        super(ALSTM, self).__init__(hidden_size =hidden_size,
                                    num_layers=num_layers,
                                    dropout=dropout,
                                    lr=lr,
                                    batch_size=batch_size,
                                    optimizer=optimizer,
                                    **kwargs)

        if gpu_util is not None:
            self.batch_size, self.hidden_size = ALSTM.GPU_UTILS_SETTINGS[gpu_util].values()
        self.set_iter()
        self.logger.info(
            "ALSTM parameters setting:"
            f"\nd_feat : {self.d_feat}"
            f"\nhidden_size : {self.hidden_size}"
            f"\nnum_layers : {self.num_layers}"
            f"\ndropout : {self.dropout}"
            f"\nlr : {self.lr}"
            f"\nbatch_size : {self.batch_size}"
            f"\noptimizer : {self.optimizer}"
            f"\ndevice : {self.device}"
            f"\nuse_GPU : {self.use_gpu}"
            f"\nseed : {self.seed}"
        )
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        self.model = ALSTMModel(
                d_feat=self.d_feat,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout)
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
        self.model.to(self.device)
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

    def train_epoch(self, x_train, y_train):
        self.model.train()
        pred = self.model(x_train)
        loss = self.loss_fn(pred, y_train)
        self.train_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
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
            with TimeEvaluator.time_context("alstm_train_epoch(no h2d copy)"):
                self.train_epoch(batch_x, batch_y)
                self.test_epoch(batch_x, batch_y)
                torch.cuda.synchronize()
            self.count_iter()
        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, input = None):
        pass

class ALSTMModel(nn.Module):
    def __init__(self, d_feat=600, hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU"):
        super().__init__()
        self.hid_size = hidden_size
        self.input_size = d_feat
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self._build_model()

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type %s" % self.rnn_type) from e
        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net.add_module("act", nn.Tanh())
        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=1)
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hid_size, out_features=int(self.hid_size / 2)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hid_size / 2), out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def forward(self, inputs):
        # inputs: [batch_size, input_size*input_day]
        inputs = inputs.view(len(inputs), self.input_size, -1)
        inputs = inputs.permute(0, 2, 1) # [batch, input_size, seq_len] -> [batch, seq_len, input_size]
        rnn_out, _ = self.rnn(self.net(inputs)) # [batch, seq_len, num_directions * hidden_size]
        attention_score = self.att_net(rnn_out) # [batch, seq_len, 1]
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        out = self.fc_out(
        torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        ) # [batch, seq_len, num_directions * hidden_si2e] -> [batch, 1]
        return out[..., 0]
