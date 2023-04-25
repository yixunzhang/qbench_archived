"""Compare speed of different models with batch size 12"""
import torch
import torch.optim as optim
import torchvision.models as models
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

torch.backends.cudnn.benchmark = True
# https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
# This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
# If you check it using the profile tool, the cnn method such as winograd, fft, etc. is used for the first iteration and the best operation is selected for the device.


MODEL_LIST = {
    models.mnasnet: ['mnasnet1_0'], 
    models.resnet: ['resnet50', 'resnet101', 'resnext50_32x4d', 'resnext101_32x8d'],
    models.densenet: ['densenet121', 'densenet201'],
    models.squeezenet: ['squeezenet1_0'],
    models.vgg: ['vgg16', 'vgg16_bn'],
    models.mobilenet: ['mobilenet_v3_large', 'mobilenet_v3_small'],
    models.shufflenetv2: ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0'],
}

precisions = ["float", "half"]
gpu_nums = [1]
# For post-voltaic architectures, there is a possibility to use tensor-core at half precision.
# Due to the gradient overflow problem, apex is recommended for practical use.
device_name = str(torch.cuda.get_device_name(0))
# Training settings
parser = argparse.ArgumentParser(description="PyTorch Benchmarking")
parser.add_argument(
    "--WARM_UP", "-w", type=int, default=5, required=False, help="Num of warm up"
)
parser.add_argument(
    "--NUM_TEST", "-n", type=int, default=50, required=False, help="Num of Test"
)
parser.add_argument(
    "--BATCH_SIZE", "-b", type=int, default=64, required=False, help="Num of batch size"
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

def train(precision, result, gpu_num):
    rand_loader = DataLoader(
        dataset=RandomDataset(args.BATCH_SIZE * gpu_num * (args.WARM_UP + args.NUM_TEST)),
        batch_size=args.BATCH_SIZE * gpu_num,
        shuffle=False,
        num_workers=8,
    )
    """use fake image for training speed test"""
    target = torch.LongTensor(args.BATCH_SIZE * gpu_num).random_(args.NUM_CLASSES).cuda()
    criterion = nn.CrossEntropyLoss()
    benchmark = {}
    for model_type in MODEL_LIST.keys():
        for model_name in MODEL_LIST[model_type]:
            model = getattr(model_type, model_name)(pretrained=False)
            if gpu_num > 1:
                model = nn.DataParallel(model, device_ids=range(gpu_num))
            model = getattr(model, precision)()
            train_optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay=1e-3)
            model = model.to("cuda")
            durations = []
            print(f"Benchmarking Training {precision} precision type {model_name} on {gpu_num} gpu")
            for step, img in enumerate(rand_loader):
                img = getattr(img, precision)()
                torch.cuda.synchronize()
                start = time.time()
                model.zero_grad()
                prediction = model(img.to("cuda"))
                loss = criterion(prediction, target)
                loss.backward()
                train_optimizer.step()
                torch.cuda.synchronize()
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
                     {psutil.cpu_freq()}\n\
                    cpu_count: {psutil.cpu_count()}\n\
                    memory_available: {psutil.virtual_memory().available}"
    gpu_configs = [
        torch.cuda.device_count(),
        torch.version.cuda,
        torch.backends.cudnn.version(),
        torch.cuda.get_device_name(0),
    ]
    gpu_configs = list(map(str, gpu_configs))
    temp = [
        "Number of GPUs on current device : ",
        "CUDA Version : ",
        "Cudnn Version : ",
        "Device Name : ",
    ]

    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    now = datetime.datetime.now()

    start_time = now.strftime("%Y/%m/%d %H:%M:%S")

    print(f"benchmark start : {start_time}")

    for idx, value in enumerate(zip(temp, gpu_configs)):
        gpu_configs[idx] = "".join(value)
        print(gpu_configs[idx])
    print(system_configs)

    with open(os.path.join(folder_name, "system_info.txt"), "w") as f:
        f.writelines(f"benchmark start : {start_time}\n")
        f.writelines("system_configs\n\n")
        f.writelines(system_configs)
        f.writelines("\ngpu_configs\n\n")
        f.writelines(s + "\n" for s in gpu_configs)

    train_result = {'type': []}
    inference_result = {'type': []}
    for gpu_num in gpu_nums:
        for precision in precisions:
            train_result['type'].append(str(gpu_num)+precision)
            train(precision, train_result, gpu_num)

    train_result_df = pd.DataFrame(train_result)
    path = f"{folder_name}/{device_name}_train_benchmark.csv"
    train_result_df.T.to_csv(path, index=True, header=False)

    now = datetime.datetime.now()

    end_time = now.strftime("%Y/%m/%d %H:%M:%S")
    print(f"benchmark end : {end_time}")
    with open(os.path.join(folder_name, "system_info.txt"), "a") as f:
        f.writelines(f"benchmark end : {end_time}\n")
