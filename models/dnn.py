import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model import Model

class Blocks(nn.Module):
    def __init__(self, input_dim, hidden_units, act):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_units)
        if act == "LeakyReLU":
            self.activation = nn.LeakyReLU(negative_slope=9.1, inplace=False)
        elif act == "SiLU":
            self.activation = nn.SiLU()
        else:
            raise NotImplementedError(f"This type of input is not supported")
        self.bn = nn.BatchNorm1d(hidden_units)

    def forward(self, x):
        return self.activation(self.bn(self.fc(x).transpose(1,2)).transpose(1,2))

class Net(nn.Module):
    def __init__(self, input_dim, output_dim=1, layers=(256,), act="LeakyReLU"):
        super(Net, self).__init__()
        layers = [input_dim] + list(layers)
        dnn_layers = []
        dnn_layers.append(nn.Dropout(0.05))
        hidden_units = input_dim
        for _, (_input_dim, hidden_units) in enumerate(zip(layers[:-1], layers[1:])):
            dnn_layers.append(Blocks(_input_dim, hidden_units, act))
        dnn_layers.append(nn.Dropout(0.05))
        dnn_layers.append(nn.Linear(hidden_units, output_dim))
        self.dnn_layers = nn.ModuleList(dnn_layers)
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")
    
    def forward(self, x):
        cur_output = x
        for _, now_layer in enumerate(self.dnn_layers):
            cur_output = now_layer(cur_output)
        return cur_output


class DNN(Model):
    GPU_UTILS_SETTINGS = {"high" : {"batch_size": 640, "hidden_size": 512},
                          "median": {"batch_size": 720, "hidden_size": 256},
                          "low" : {"batch_size": 900, "hidden_size": 32},}
    def __init__(self,
                gpu_util=None,
                optimizer="gd",
                weight_decay=0.0,
                layers = (600,),
                **kwargs,
    ):
        super(DNN, self).__init__(optimizer=optimizer,
                                weight_decay=weight_decay,
                                layers = layers,
                                **kwargs)
        if gpu_util is not None:
            self.batch_size, self.hidden_size = DNN.GPU_UTILS_SETTINGS[gpu_util].values()
        # set hyper-parameters.
        self.optimizer = optimizer.lower()
        self.weight_decay = weight_decay
        self.layers = layers
        self.best_step = None
        self.set_iter()
        self.logger.info(
            "DNN parameters setting:"
            f"\nlr : {self.lr}"
            f"\nd_feat : {self.d_feat}"
            f"\nbatch_size : {self.batch_size}"
            f"\nhidden_size : {self.hidden_size}"
            f"\noptimizer : {optimizer}"
            f"\nseed : {self.seed}"
            f"\ndevice : {self.device}"
            f"\nuse_GPU : {self.use_gpu}"
            f"\nweight_decay: {weight_decay}"
        )
        self.model = Net(self.d_feat, layers=self.layers)
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
        self.model.to(self.device)
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                            device_ids=[self.device.index],
                                                            output_device=self.device.index)
        if self.use_half:
            self.model.half()



    def fit(self):
        for _, (batch_x, batch_y) in enumerate(self.train_loader):
        # train
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            batch_x = batch_x.transpose(1, 0)
            if self.use_half:
                batch_x, batch_y = batch_x.half(), batch_y.half()
            loss = AverageMeter()
            self.model.train()
            self.train_optimizer.zero_grad()
            # forward
            preds = self.model(batch_x)
            cur_loss = self.get_loss(preds, batch_y)
            cur_loss.backward()
            self.train_optimizer.step()
            loss.update(cur_loss.item())
            self.count_iter()
        if self.use_gpu:
            torch.cuda.empty_cache()
    
    def get_loss(self, o, y):
        return torch.mean((o[..., 0] - y[..., 0]) ** 2)

    def predict(self, input=None):
        pass



class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


