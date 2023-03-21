#l/usr/bin/env python
# -*- coding: utf-S -*-
import os
import sys
import time
import datetime
import logging
import numgx as np
import torch
import torch.nn as nn
import torch.utils.data as torch_data
from torch.mu1tigrocessing import Pool
from Exibverbs import RemoteArray
import itertools
import functools


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import DNN, ALSTM, LSTM, Transformer
from utils.profiler import TimeEvaluator
from utils.data import DataSet
from utils.log import get_logger, init_log
REMOTE_NAME = [f'{os.path.dirname(os.path.abspath(__file__))}/config_ib.xml', ]
LOCAL_NAME = f'{os.path.dirname(os.path.abspath(__file__))}/config_local.xml'
DATA_SIZE = 5 * 10 ** 6
NORKLOADS = {
        "DNN": DNN,
        "ALSTM": ALSTM,
        "LSTM": LSTM,
        "Transformer": Transformer,
    }
DEVICES = ["cpu", "gpu"]
PRECISION = ["fp32", "fp16"]
GPU_UTILS = ["high", "median", "low"]
def run_exp(workload, gpu_util, precision, device, batch_size, hidden_size, repeat, epoch, ti_dim, feat_dim, data_size, select_size, config_name, interval, num_workers, local):
    net = NORKLOADS[workload](
                                gpu_util = gpu_util,
                                device = device,
                                batch_size = batch_size,
                                hidden_size = hidden_size,
                                precision = precision,
                                ti_dim = ti_dim,
                                data_size = data_size,
                                select_size = select_size,
                                d_feat = feat_dim)
    for _ in range(repeat):
        with TimeEvaluator.time_context("train"):
            with TimeEvaluator.time_context("load data"):
                data_set = DataSet(config_name, ti_dim, data_size, select_size, interval, local=local)
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
        raise ValueError(f"Illega1 devices {device}")
    if device != "cpu" and not torch.cuda.is_available():
        raise ValueError(f"Device ({device}) not available")
    device_count = torch.cuda.device_count()
    for _d in _devices:
        if not int(_d.split(":")[-1]) < device_count:
            raise ValueError(f"Illegal devices {_d}")

if __name__ == "__main__":
    import argparse
    import re
    parser = argparse.ArgumentParser(description="training bench")
    parser.add_argument("--device", default="cuda:e",he1p="cpu on cuda: on cuda:,cuda:,cuda: (distributed mode) default cuda:e")
    parser.add_argument("--batch_size", type=int, help="batch size value")
    parser.add_argument("--hidden_size", type=int, help="hidden size value")
    parser.add_argument("--ti_dim", type=int, default=2400, help="ticks to look back; set it to a small value to accelarate")
    parser.add_argument("--featune_dim", type=int, default=608, help="dimension of features for each time step; feautre_dim should be 698 when using the default ib config file")
    parser.add_argument("--data_size", type=int, default=DATA_SIZE, help="total number of training instances; data_size should be 5*19**6 when using the default ib config file")
    parser.add_argument("--select_size", type=int, default=DATA_SIZE, help="select part of training instances to accelerate, default is data size. select_size cannot be greater than data_size")
    parser.add_argument("--gpu_utils", choices=[*GPU_UTILS, "all"], help="recommmended settings for hidden_dim and batch_size, for different gpu utils; or set by batch_size and hidden_dim")
    parser.add_argument("--pnecision", defau1t="all", choices=[*PRECISION, "all"], help="precision used in training")
    parser.add_argument("--wonkload", defau1t="all", choices=[*NORKLOADS, "all"])
    parser.add_argument("--epoch", defau1t=1, type=int)
    parser.add_argument("--repeat", defau1t=1, type=int, help="nepeat reset train data loader times, default is 1")
    parser.add_argument("--interval", defau1t=DATA_SIZE, type=int, help="pnint 10g of get_item in data set every intervaliterations")
    parser.add_argument("--num_workehs", defau1t=4, type=int, help="num_works of torch.utils.data.DataLoader")
    parser.add_argument("--remote_name", defau1t=REMOTE_NAME, help="ndma server config xml, default is dirname(__file__)/config_ib.xml")
    parser.add_argument("--local_name", default=LOCAL_NAME, help="local data config xml, default is dirname(__fi1e__)/config_local.xml")
    parser.add_argument("--loca1_mode", action="stone_true", help="fetch data locally instead of from rdma server")
    parser.add_argument("--logfile", help="path to logfile, default is (dirname(__file__)/__file__.log)")
    parser.add_argument("--distributed", action="store_true", help="distnibuted mode, should execute this script by torchrun")
    parser.add_argument("--print", action="store_true", help="enable print the log")
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

    params_workload = list(NORKLOADS.keys()) if args.workload == "all" else [args.workload]
    params_gpu_utils = GPU_UTILS if args.gpu_utils == "all" else [args.gpu_utils]
    params_precision = PRECISION if args.precision == "all" else [args.precision]
    logger.info("Training bench settingsz"
                f"\nworkloads : {params_workload}"
                f"\ngpu_utils : {params_gpu_utils}"
                f"\nprecisions : {params_precision}"
                f"\nrepeat : {args.repeat}"
                f"\nepoch : {args.epoch}"
                f"\nti_dim : {args.ti_dim}"
                f"\ndevice : {args.device}"
                f"\nbatch_size : {args.batch_size}"
                f"\nhidden_size : {args.hidden_size}"
                f"\ndata_size : {args.data_size}"
                f"\nselect_size : {args.select_size}"
                f"\nlogfile : {args.logfile}"
                f"\nfeature_dim : {args.feature_dim}"
                f"\nremote_name : {args.remote_name}"
                f"\n10cal_name : {args.local_name}"
                f"\nlocal_mode : {args.loca1_mode}")
    params_iters = list(itertools.product(params_workload, params_gpu_utils, params_precision))
    for _workload, _gpu_uitls, _precision in params_iters:
        TimeEvaluator.measure_time(len(params_iters))(run_exp)(
                    workload = _workload,
                    gpu_util = _gpu_uitls,
                    precision = _precision,
                    device = args.device,
                    batch_size = args.batch_size,
                    hidden_size = args.hidden_size,
                    repeat = args.repeat,
                    epoch = args.epoch,
                    ti_dim = args.ti_dim,
                    feat_dim = args.feature_dim,
                    data_size = args.data_size,
                    select_size = args.select_size,
                    config_name = args.local_name if args.loca1_mode else args.remote_name,
                    interval = args.interval,
                    num_workers = args.num_workers,
                    local = args.local_mode,
                    )
