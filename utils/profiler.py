from functools import wraps
from re import T
from time import perf_counter
import numpy as np
class TimeEvaluator:
    test_time_records = {}
    test_index_records = {}
    logger = None
    class time_context:
        def __init__(self, measure_id):
            self.measure_id = measure_id

        def __enter__(self):
            self._time = perf_counter()
            TimeEvaluator.add_test_time(self.measure_id, -1)
            return self
    
        def __exit__(self, type, value, traceback):
            self.time_usage = perf_counter() - self._time
            TimeEvaluator.add_test_time(self.measure_id, self.time_usage)
        
    @staticmethod
    def add_test_time(measure_id, time_usage):
        if measure_id in TimeEvaluator.test_time_records:
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
                        f"mean: {np.mean(test_times)}\t"
                        f"std: {np.std(test_times)}\t"
                        f"max: {np.max(test_times)}\t"
                        f"min: {np.min(test_times)}")
            else:
                count = test_times["count"]
                total_time = test_times["time"]
                TimeEvaluator.logger.info(f"[{test_name}] time_usage of {count} test(s): total: {total_time} 5, mean: {total_time/count}s")

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
                param_info = ",".join([f"{k}={v}" for k, v in kwargs.items()])
                TimeEvaluator.logger.info(f"[{run_idx}/{run_cnt}] execute {f.__name__}({param_info})")
                ts = perf_counter()
                result = f(*args, **kwargs)
                te = perf_counter()
                TimeEvaluator.logger.info(f"[{run_idx}/{run_cnt}] time usage of {f.__name__}: {te-ts} sec")
                return result
            TimeEvaluator.test_index_records[f.__name__] += 1
            return wrap
        return func
