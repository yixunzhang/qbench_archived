from functools import wraps
from re import T
from time import perf_counter
import numpy as np
import torch
import pandas as pd
import os
import re

class TimeEvaluator:
    test_time_records = {}
    test_index_records = {}
    df_buffer = None
    file_name = None
    logger = None
    test_index_records = {}
    class time_context:
        def __init__(self, measure_id, warmup=0):
            self.measure_id = measure_id
            self.warmup = warmup
            if self.measure_id in TimeEvaluator.test_index_records:
                TimeEvaluator.test_index_records[self.measure_id] += 1
            else:
                TimeEvaluator.test_index_records[self.measure_id] = 0

        def __enter__(self):
            if TimeEvaluator.test_index_records[self.measure_id] < self.warmup:
                return self
            self._time = perf_counter()
            TimeEvaluator.add_test_time(self.measure_id, -1)
            return self
    
        def __exit__(self, type, value, traceback):
            if TimeEvaluator.test_index_records[self.measure_id] < self.warmup:
                return
            self.time_usage = perf_counter() - self._time
            TimeEvaluator.add_test_time(self.measure_id, self.time_usage)
        
    @staticmethod
    def add_test_time(measure_id, time_usage):
        if measure_id in TimeEvaluator.test_time_records:
            if time_usage < 0:
                return
            if isinstance(TimeEvaluator.test_time_records[measure_id], list):
                if len(TimeEvaluator.test_time_records[measure_id]) > 10**3:
                    times = TimeEvaluator.test_time_records[measure_id]
                    TimeEvaluator.test_time_records[measure_id] = {"time":np.sum(times) + time_usage, "count":len(times)}
                else:
                    TimeEvaluator.test_time_records[measure_id].append(time_usage)
            else: # dict
                TimeEvaluator.test_time_records[measure_id]["time"] += time_usage
                TimeEvaluator.test_time_records[measure_id]["count"] += 1
        elif time_usage < 0:
            TimeEvaluator.test_time_records[measure_id] = []

    @staticmethod
    def get_info(measure_id = None):
        if measure_id is None:
            items = TimeEvaluator.test_time_records.items()
        elif measure_id not in TimeEvaluator.test_time_records:
            return
        else:
            items = [(measure_id, TimeEvaluator.test_time_records[measure_id])]
        for test_name, test_times in items:
            if isinstance(test_times, list):
                TimeEvaluator.logger.info(
                        f"[{test_name}] time_usage of {len(test_times)} test(s): "
                        f"sum: {np.sum(test_times)}\t"
                        f"mean:{np.mean(test_times)}\t"
                        f"std: {np.std(test_times)}\t"
                        f"max: {np.max(test_times)}\t"
                        f"min: {np.min(test_times)}")
                TimeEvaluator.dump_to_csv({test_name+"(ms)" : [np.mean(test_times)*1000]})
            else:
                count = test_times["count"]
                total_time = test_times["time"]
                TimeEvaluator.logger.info(f"[{test_name}] time_usage of {count} test(s): total: {total_time} 5, mean: {total_time/count}s")
                TimeEvaluator.dump_to_csv({test_name+"(ms)" : [total_time/count * 1000]})

    @staticmethod
    def clear_records():
        TimeEvaluator.test_time_records.clear()
    
    @staticmethod
    def set_log(logger):
        TimeEvaluator.logger = logger
    
    @staticmethod
    def measure_time(run_cnt=1):
        def func(f):
            if f.__name__ not in TimeEvaluator.test_index_records:
                TimeEvaluator.test_index_records[f.__name__] = 0
            run_idx = TimeEvaluator.test_index_records[f.__name__]
            @wraps (f)
            def wrap(*args, **kwargs):
                TimeEvaluator.init_csv_dump_info(kwargs)
                test_on_gpu = kwargs["device"] != "cpu"
                param_info = ",".join([f"{k}={v}" for k, v in kwargs.items()])
                TimeEvaluator.logger.info(f"[{run_idx}/{run_cnt}] execute {f.__name__}({param_info})")
                ts = perf_counter()
                result = f(*args, **kwargs)
                if test_on_gpu:
                    torch.cuda.synchronize()
                te = perf_counter()
                TimeEvaluator.logger.info(f"[{run_idx}/{run_cnt}] time usage of {f.__name__}: {te-ts} sec")
                TimeEvaluator.dump_to_csv({"overall-time(s)" : [te-ts]}, dump=True)
                return result
            TimeEvaluator.test_index_records[f.__name__] += 1
            return wrap
        return func
    
    @staticmethod
    def init_csv_dump_info(kwargs):
        params = {k:[v] for k, v in kwargs.items() if k in ["workload", "gpu_util", "precision", "measure"]}
        if TimeEvaluator.file_name is None:
            if kwargs["csv_file"] is None:
                if kwargs["device"] == "cpu":
                    import cpuinfo
                    device_info = cpuinfo.get_cpu_info()["brand_raw"]
                else:
                    device_info = torch.cuda.get_device_name(kwargs["device"].split(",")[0])
                file_name = re.sub('[^a-zA-Z0-9]', '_', device_info.replace(" ", "")) + ".csv"
                TimeEvaluator.file_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "result", file_name)
            else:
                TimeEvaluator.file_name = kwargs["csv_file"]
            if os.path.exists(TimeEvaluator.file_name):
                os.remove(TimeEvaluator.file_name)
        TimeEvaluator.df_buffer = pd.DataFrame(params)

    @staticmethod
    def dump_to_csv(test_data, dump = False):
        key = list(test_data.keys())[0]
        if "train_epoch" in key:
            test_data = {key[key.find("_")+1:]:test_data[key]}
        df_curr = pd.DataFrame(test_data)
        TimeEvaluator.df_buffer = pd.concat([TimeEvaluator.df_buffer, df_curr], axis=1)
        if dump:
            if os.path.exists(TimeEvaluator.file_name):
                df_prev = pd.read_csv(TimeEvaluator.file_name)
                df = pd.concat([df_prev, TimeEvaluator.df_buffer], axis=0)
                df.to_csv(TimeEvaluator.file_name, index=False)
            else:
                TimeEvaluator.df_buffer.to_csv(TimeEvaluator.file_name, index=False)
            TimeEvaluator.df_buffer = None
