#l/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import datetime
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as torch_data
from torch.multiprocessing import Pool
import itertools
import functools
import pandas as pd
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import DNN, ALSTM, LSTM, Transformer, ResNet50
from utils.profiler import TimeEvaluator
from utils.data import DataSet
from utils.log import get_logger, init_log
from utils.localdata import check_dataset, generate_random_dataset
REMOTE_NAME = [f'{os.path.dirname(os.path.abspath(__file__))}/config_ib.xml', ]
LOCAL_NAME = f'{os.path.dirname(os.path.abspath(__file__))}/config_local.xml'
DATA_SIZE = 5 * 10 ** 6
WORKLOADS = {
        "DNN": DNN,
        "ALSTM": ALSTM,
        "LSTM": LSTM,
        "Transformer": Transformer,
        "ResNet50": ResNet50,
    }
PRECISION = ["fp32", "fp16", "bp16"]
PRECISION_GPU = ["fp32", "fp16"]
PRECISION_CPU = ["fp32", "bf16"]
GPU_UTILS = ["high", "median", "low"]
MEASURE_TYPE = ["end2end", "no_h2d"]

def run_exp(workload, gpu_util, precision, measure, device, batch_size, hidden_size, repeat, epoch, ti_dim, feat_dim, data_size, max_iters, config_name, interval, num_workers, local, csv_file):
    net = WORKLOADS[workload](
                                gpu_util = gpu_util,
                                device = device,
                                batch_size = batch_size,
                                hidden_size = hidden_size,
                                precision = precision,
                                ti_dim = ti_dim,
                                data_size = data_size,
                                max_iters = max_iters,
                                d_feat = feat_dim, 
                                meaure = measure)
    for _ in range(repeat):
        with TimeEvaluator.time_context("train-overall"):
            with TimeEvaluator.time_context("init data"):
                data_set = DataSet(config_name, ti_dim, data_size, interval, local=local)
                net.train_loader_init(data_set, num_workers=num_workers, local=local)
            for _ in range(epoch):
                net.fit()
    TimeEvaluator.get_info()
    TimeEvaluator.clear_records()

def check_device(device):
    _devices = device.split(",")
    if len(_devices) <= 1:
        if re.match("cpu", device) is None and re.match("cuda:\d+$", device) is None:
            raise ValueError(f"Illegal device {device}")
    elif not functools.reduce(lambda x, y : x and re.match("cuda:\d+$", y) is not None, _devices):
        raise ValueError(f"Illegl devices {device}")
    if device != "cpu" and not torch.cuda.is_available():
        raise ValueError(f"Device ({device}) not available")
    if device != "cpu":
        device_count = torch.cuda.device_count()
        for _d in _devices:
            if not int(_d.split(":")[-1]) < device_count:
                raise ValueError(f"Illegal devices {_d}")

def analyze_result(path):
    df_test=pd.read_csv(path)
    df_base = pd.read_csv(f"{os.path.dirname(os.path.abspath(__file__))}/result/baseline_private.csv")
    models=set(df_base["workload"])
    precisions = set(df_base["precision"])
    measures = set(df_base["measure"]) 

    df_acc = df_test
    df_acc["train_epoch(ms)"] = df_base["train_epoch(ms)"]  / df_test["train_epoch(ms)"]
    labels=[]
    boxes=[]
    for precision in precisions:
        for measure in measures:
            values = df_acc.loc[df_acc["precision"]==precision].loc[df_base["measure"]==measure]
            boxes.append(np.concatenate([values[values["workload"]==model]["train_epoch(ms)"].values for model in models])) 
            labels.append("-".join([precision, measure]))
    bp=plt.boxplot(boxes, showmeans=True, meanline=True, labels=labels)
    plt.savefig(path.split(".")[0]+"_summary.png")
    mean_vals={k:m.get_ydata()[0] for k, m in zip(labels, bp["means"])}
    median_vals = {k:m.get_ydata()[0] for k, m in zip(labels, bp["medians"])}
    whiskers_vals =list(map(lambda x:[*x[0],*x[1]], zip(*[iter([m.get_ydata() for m in bp["whiskers"]])]*2)))
    upper_quartile_vals = {k:v[0] for k, v in zip(labels, whiskers_vals)}
    lower_quartile_vals = {k:v[2] for k, v in zip(labels, whiskers_vals)}
    min_vals= {k:v[1] for k, v in zip(labels, whiskers_vals)}
    max_vals ={k:v[3] for k, v in zip(labels, whiskers_vals)}
    summary = {"min" : min_vals,
               "upper_quartile": upper_quartile_vals,
               "median" : median_vals,
               "mean":mean_vals,
               "lower_quartile":lower_quartile_vals,
               "max" : max_vals}
    print(pd.DataFrame(summary))
    print(＂首行解释：在所跑的模型中，在精度{}，{}数据(end2end包含了数据拷贝时间，no_h2d表示不包含数据拷贝时间)，最小加速为{:.4f}倍，75％了至少{:.4f}倍加速，50％获取了至少{:.4f}倍加速，平均获得{:.4f}倍加速，25％获取了至少{:.4f}倍加速，最大加速值为{:.4f}倍.format(
            labels[0].split("-")[0], labels[0].split("-")[1], min_vals[labels[0]], upper_quartile_vals[labels[0]],
            median_vals[labels[0]], mean_vals[labels[0]],lower_quartile_vals[labels[0]], max_vals[labels[0]]))
    summary_path = path.split(".")[0] + "_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path) 
    print(f"summary saved in {summary_path}") 

if __name__ == "__main__":
    import argparse
    import re
    parser = argparse.ArgumentParser(description="training bench")
    parser.add_argument("--device", default="cuda:0",help="cpu or cuda:<device_id> or cuda:<device_id>,cuda:<deivce_id>,cuda:<device_id> (distributed mode) default cuda:0")
    parser.add_argument("--batch_size", type=int, help="batch size value")
    parser.add_argument("--hidden_size", type=int, help="hidden size value")
    parser.add_argument("--ti_dim", type=int, default=2400, help="ticks to look back; set it to a small value to accelarate")
    parser.add_argument("--feature_dim", type=int, default=600, help="dimension of features for each time step; feautre_dim should be 698 when using the default ib config file")
    parser.add_argument("--data_size", type=int, default=20000, help="total number of training instances; data_size should be 5*19**6 when using the default ib config file")
    parser.add_argument("--max_iters", type=int, default=20, help="the limit of train epoch times")
    parser.add_argument("--gpu_utils", default="all", choices=[*GPU_UTILS, "all"], help="recommmended settings for hidden_dim and batch_size, for different gpu utils; or set by batch_size and hidden_dim")
    parser.add_argument("--precision", default="all", choices=[*PRECISION, "all"], help="precision used in training, fp16 only available on cuda and bf16 only available on cpu")
    parser.add_argument("--workload", default="all", choices=[*WORKLOADS, "all"])
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--repeat", default=1, type=int, help="nepeat reset train data loader times, default is 1")
    parser.add_argument("--interval", default=DATA_SIZE, type=int, help="print 10g of get_item in data set every intervaliterations")
    parser.add_argument("--num_workers", default=4, type=int, help="num_works of torch.utils.data.DataLoader")
    parser.add_argument("--remote_name", default=REMOTE_NAME, help="ndma server config xml, default is dirname(__file__)/config_ib.xml")
    parser.add_argument("--local_name", default=LOCAL_NAME, help="local data config xml, default is dirname(__file__)/config_local.xml")
    parser.add_argument("--local_mode", default=True, type=bool, help="fetch data locally instead of from rdma server")
    parser.add_argument("--logfile", help="path to logfile, default is (dirname(__file__)/__file__.log)")
    parser.add_argument("--distributed", action="store_true", help="distnibuted mode, should execute this script by torchrun")
    parser.add_argument("--print", default=True, type=bool, help="enable print the log")
    parser.add_argument("--csv_file", help="path to csv results, default is dirname(__file__)/result/<device_name>.csv")
    parser.add_argument("--measure", default="all", choices=[*MEASURE_TYPE, "all"], help="measure train time with h2d copy(end2end) or without h2d copy(no_h2d)")
    args = parser.parse_args()
    logpath = __file__[:__file__.rfind('.')] + datetime.datetime.now().strftime("_%Y%m%d_%H_%M_%S") + ".log" if args.logfile is None else args.logfile
    logger = init_log(__name__, level=logging.INFO, logpath=logpath, enable_print=args.print)
    logger = get_logger()
    TimeEvaluator.set_log(logger)
    # Check params
    if args.distributed:
        torch.distributed.init_process_group(backend="nccl")
    check_device(args.device)
    logger.info(f"set device: {args.device}")

    params_workload = list(WORKLOADS.keys()) if args.workload == "all" else [args.workload]
    params_gpu_utils = GPU_UTILS if args.gpu_utils == "all" else [args.gpu_utils]
    _all_precision = PRECISION_GPU if args.device.startswith("cuda") else PRECISION_CPU
    params_precision = _all_precision if args.precision == "all" else [args.precision]
    params_measure = MEASURE_TYPE if args.measure == "all" else [args.measure]
    logger.info("Training bench raw settings"
                f"\nworkloads   : {params_workload}"
                f"\ngpu_utils   : {params_gpu_utils}"
                f"\nprecisions  : {params_precision}"
                f"\nrepeat      : {args.repeat}"
                f"\nepoch       : {args.epoch}"
                f"\nti_dim      : {args.ti_dim}"
                f"\ndevice      : {args.device}"
                f"\nbatch_size  : {args.batch_size}"
                f"\nhidden_size : {args.hidden_size}"
                f"\ndata_size   : {args.data_size}"
                f"\nmax_iters   : {args.max_iters}"
                f"\nlogfile     : {logpath}"
                f"\nfeature_dim : {args.feature_dim}"
                f"\nremote_name : {args.remote_name}"
                f"\nlocal_name  : {args.local_name}"
                f"\nlocal_mode  : {args.local_mode}")
    # check data set
    if not check_dataset(args.local_name):
        generate_random_dataset(args.local_name)
    params_iters = list(itertools.product(params_workload, params_gpu_utils, params_precision, params_measure))
    for _workload, _gpu_uitls, _precision, _measure in params_iters:
        if args.device == "cpu" and _precision == "fp16":
            continue
        if args.device != "cpu" and _precision == "bf16":
            continue
        TimeEvaluator.measure_time(len(params_iters))(run_exp)(
                    workload = _workload,
                    gpu_util = _gpu_uitls,
                    precision = _precision,
                    measure = _measure,
                    device = args.device,
                    batch_size = args.batch_size,
                    hidden_size = args.hidden_size,
                    repeat = args.repeat,
                    epoch = args.epoch,
                    ti_dim = args.ti_dim,
                    feat_dim = args.feature_dim,
                    data_size = args.data_size,
                    max_iters = args.max_iters,
                    config_name = args.local_name if args.local_mode else args.remote_name,
                    interval = args.interval,
                    num_workers = args.num_workers,
                    local = args.local_mode,
                    csv_file = args.csv_file
                    )
    analyze_result(TimeEvaluator.file_name)
