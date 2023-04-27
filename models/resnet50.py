import torch
import torch.nn as  nn
import torch.optim as optim
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model import Model
from utils.profiler import TimeEvaluator

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        return x

class ResNet50Model(nn.Module):
    def __init__(self, layer_list=[3, 4, 6, 3], num_classes=100, num_channels=3):
        super(ResNet50Model, self).__init__()
        self.in_channels = 64
        ResBlock = Bottleneck

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion

        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


class ResNet50(Model):
    gpu_utils = {
            "low":{"batch_size":32},
            "median":{"batch_size":64},
            "high":{"batch_size":128},}
    def __init__(self, gpu_util=None, **kwargs):
        super(ResNet50, self).__init__(**kwargs)
        if gpu_util is not None:
            self.batch_size = ResNet50.gpu_utils[gpu_util]["batch_size"]
        self.set_iter()
        self.logger.info(
                "ResNet50 parameters setting:"
                f"\nbatch_size: {self.batch_size}"
                f"\nuse_GPU:    {self.use_gpu}"
                f"\nseed:       {self.seed}"
        )
        self.model = ResNet50Model(num_channels=self.batch_size)
        self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        
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
        self.train_optimizer.step()
    
    def test_epoch(self, batch_x, batch_y):
        self.model.eval()
        pred = self.model(batch_x)
        loss = self.loss_fn(pred, batch_y)
        score = self.metric_fn(pred, batch_y)
        return loss, score

    def fit(self):
        for _, (batch_x, batch_y) in enumerate(self.train_loader):
            with TimeEvaluator.time_context("resnet50_train_epoch", warmup=5):
                if batch_x.shape[0] != self.batch_size:
                    self.count_iter()
                    continue
                if self.use_gpu:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                if self.use_half:
                    batch_x, batch_y = batch_x.half(), batch_y.half()
                # train
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
