# qbench
pytorch training benchmark

## Generate test data
1. Modify the `target_file` and `data_shape` in `config_local.xml`
2. Run utils/localdata.py by executing `python3 utils/localdata.py -c config_local.xml`

## Run test
1. `python3 train.py --local_mode --print --workload LSTM --gpu_utils low --precision fp32 --select_size 5000 --data_size 500000`
2. Check details by `python train.py --help`
