"""Compare speed of different models with batch size 12"""
import torch
import torch.optim as optim
import torchvision.models as models
import intel_extension_for_pytorch as ipex
import platform
import psutil
import torch.nn as nn
import datetime
import time
import os
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
import json

MODEL_LIST = {
    models.mnasnet: ['mnasnet1_0'], 
    models.resnet: ['resnet50', 'resnet101', 'resnext50_32x4d', 'resnext101_32x8d'],
    models.densenet: ['densenet121', 'densenet201'],
    models.squeezenet: ['squeezenet1_0'],
    models.vgg: ['vgg16', 'vgg16_bn'],
    models.shufflenetv2: ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0'],
}

precisions = ["float", "bfloat16"]
# Training settings
parser = argparse.ArgumentParser(description="PyTorch Benchmarking")
parser.add_argument(
    "--WARM_UP", "-w", type=int, default=5, required=False, help="Num of warm up"
)
parser.add_argument(
    "--NUM_TEST", "-n", type=int, default=50, required=False, help="Num of Test"
)
parser.add_argument(
    "--BATCH_SIZE", "-b", type=int, default=12, required=False, help="Num of batch size"
)
parser.add_argument(
    "--NUM_CLASSES", "-c", type=int, default=1000, required=False, help="Num of class"
)
parser.add_argument(
    "--folder",
    "-f",
    type=str,
    default="result",
    required=False,
    help="folder to save results",
)
args = parser.parse_args()

class RandomDataset(Dataset):
    def __init__(self, length):
        self.len = length
        self.data = torch.randn(3, 224, 224, length)

    def __getitem__(self, index):
        return self.data[:, :, :, index]

    def __len__(self):
        return self.len

def train(precision, result, use_opt):
    rand_loader = DataLoader(
        dataset=RandomDataset(args.BATCH_SIZE * (args.WARM_UP + args.NUM_TEST)),
        batch_size=args.BATCH_SIZE,
        shuffle=False,
        num_workers=8,
    )
    """use fake image for training speed test"""
    target = torch.LongTensor(args.BATCH_SIZE).random_(args.NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    benchmark = {}
    for model_type in MODEL_LIST.keys():
        for model_name in MODEL_LIST[model_type]:
            model = getattr(model_type, model_name)(pretrained=False)
            if not use_opt:
                model = getattr(model, precision)()
            train_optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay=1e-3)
            if use_opt:
                model, train_optimizer = ipex.optimize(model, optimizer=train_optimizer, dtype=torch.bfloat16 if precision=="bfloat16" else torch.float)
            durations = []
            print(f"Benchmarking Training {precision} precision type {model_name} with opt={use_opt}")
            for step, img in enumerate(rand_loader):
                img = getattr(img, precision)()
                start = time.time()
                model.zero_grad()
                if precision == "bfloat16":
                    with torch.cpu.amp.autocast():
                        prediction = model(img)
                        loss = criterion(prediction, target)
                        loss.backward()
                else:
                    prediction = model(img)
                    loss = criterion(prediction, target)
                    loss.backward()
                train_optimizer.step()
                end = time.time()
                if step >= args.WARM_UP:
                    durations.append((end - start) * 1000)
            print(
                f"{model_name} model average train time : {sum(durations)/len(durations)}ms"
            )
            del model
            if model_name not in result.keys():
                result[model_name] = [sum(durations)/len(durations)]#durations
            else:
                result[model_name].append(sum(durations)/len(durations))

f"{platform.uname()}\n{psutil.cpu_freq()}\ncpu_count: {psutil.cpu_count()}\nmemory_available: {psutil.virtual_memory().available}"

if __name__ == "__main__":
    folder_name = args.folder

    system_configs = f"{platform.uname()}\n\
                       {platform.processor()}"

    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    now = datetime.datetime.now()

    start_time = now.strftime("%Y/%m/%d %H:%M:%S")

    print(f"benchmark start : {start_time}")

    print(system_configs)

    with open(os.path.join(folder_name, "system_info.txt"), "w") as f:
        f.writelines(f"benchmark start : {start_time}\n")
        f.writelines("system_configs\n\n")
        f.writelines(system_configs)

    train_result = {'type': []}
    for precision in precisions:
        if precision != "bfloat16":
            train_result['type'].append("torch " + precision)
            train(precision, train_result, False)
        train_result['type'].append("intel ipex " + precision)
        train(precision, train_result, True)

    train_result_df = pd.DataFrame(train_result)
    path = f"{folder_name}/cpu_train_benchmark.csv"
    train_result_df.T.to_csv(path, index=True, header=False)

    now = datetime.datetime.now()

    end_time = now.strftime("%Y/%m/%d %H:%M:%S")
    print(f"benchmark end : {end_time}")
    with open(os.path.join(folder_name, "system_info.txt"), "a") as f:
        f.writelines(f"benchmark end : {end_time}\n")
