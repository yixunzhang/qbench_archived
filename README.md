# qbench
pytorch training benchmark

## Generate test data
1. Modify the `target_file` and `data_shape` in `config_local.xml`
2. Run utils/localdata.py by executing `python3 utils/localdata.py -c config_local.xml`

## Run test
1. For single GPU: `python3 train.py --local_mode --print --workload LSTM --gpu_utils low --precision fp32 --select_size 5000 --data_size 500000`
2. For multiple GPUs:  `torchrun --nproc_per_node 4 train.py --local_mode --print --workload ALSTM --precision fp32 --gpu_utils low --data_size 500000 --select_size 50000 --distributed --device cuda:0,cuda:1,cuda:2,cuda:3`
3. Check details by `python train.py --help`

## Docker images
1. For Nvdia GPU: 
    - `docker pull johnzlli/qbench:pytorch1.12.1-cu113`
    - `docker run --runtime=nvidia -it --rm --shm-size 50G johnzlli/qbench:pytorch1.12.1-cu113 /bin/bash`
2. For AMD GPU:
    - `docker pull johnzlli/qbench:pytorch1.13.0-rocm5.4`
    - `docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined johnzlli/qbench:pytorch1.13.0-rocm5.4`
