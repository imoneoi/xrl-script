# XRL-Script

Efficient distributed AutoRL script for any framework

## Features

- [x] Distributed training on single machine, multiple CPU/GPUs
- [x] Automatic resource allocation
- [x] TPESampler and Hyperband pruner support
- [ ] Automatic testing for stable hyperparameter set

## Usage

*In rl.py*
1. Implement your training logic in train_rl_agent
2. Specify hyperparameter range

*In rl-auto-gpu.py*
1. Specify minimum and maximum steps per trial, and number of trials
2. Set parallel trials per GPU & parallel envs per trial according to your hardware
3. Set reduction factor (integer). In most situations you can set one so that Hyperband bracket number stays in [4, 6]

**Then run rl-auto-gpu.py, the script will find all available gpus and run hyperparameter search in parallel**

## Dependencies

- [Optuna](https://github.com/optuna/optuna)