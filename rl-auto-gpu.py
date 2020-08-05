from .rl import train_rl_agent

import numpy as np

import os
from os import path
from datetime import datetime
import time
import json
import subprocess
from threading import Lock
from typing import Union, Callable, Optional
from multiprocessing import Process, Pipe

import optuna

def train_rl_agent_worker(pipe, hyperparameters):
    try:
        try:
            #should prune
            def prune_func(train_reward, env_step):
                pipe.send(["shouldprune", {
                    "rew": train_reward,
                    "step": env_step
                }])
                is_prune = pipe.recv()
                if is_prune:
                    raise optuna.TrialPruned()

            #train model
            final_reward = train_rl_agent(
                prune_func,
                **hyperparameters
            )

            pipe.send(["end", final_reward])
        except optuna.TrialPruned:
            pipe.send(["pruned", None])
        except Exception as e:
            pipe.send(["error", e])
    finally:
        pipe.close()

def find_available_gpus(util_threshold=20, mem_threshold=10, test_time=10):
    gpu_isbusy = {}
    for _ in range(test_time):
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')
        gpu_info_list = [[float(item.strip()) for item in line.split(',')] for line in result.strip().split('\n')]

        for id, gpu_info in enumerate(gpu_info_list):
            #push id into dict
            if not id in gpu_isbusy:
                gpu_isbusy[id] = False

            #check if busy
            gpu_util = gpu_info[0]
            gpu_mem  = gpu_info[1] / gpu_info[2]

            print("GPU {} Utilization {:.2f}% Mem {:.2f}%".format(id, gpu_util, gpu_mem))
            if not ((gpu_util < util_threshold) and (gpu_mem < mem_threshold)):
                gpu_isbusy[id] = True

        time.sleep(0.1)

    available_gpus = ["cuda:" + str(id) for id, busy in gpu_isbusy.items() if not busy]

    print("Available GPUs: " + str(available_gpus))
    return available_gpus

def train_rl_agent_auto(
    #Env config
    env_name: str = "LunarLanderContinuous-v2",
    env_reward_threshold: Optional[int] = 200,

    #Hyperband config
    max_env_steps: int = 1e6,
    min_trial_env_steps: int = 1e5,
    reduction_factor: float = 2,

    #Number of trials
    num_trials: Optional[int] = None,

    #Thread per GPU
    num_threads_per_gpu: int = 4,

    #Env per thread
    parallel_envs: int = 8,

    #Logdir
    logdir: str = "logs/autotune_with_ans_{}".format(datetime.now().strftime("%m-%d_%H-%M-%S"))
):
    #GPU usage statistics
    global_lock = Lock()
    global_num_gpu_threads = {}

    #Get free GPUs
    gpu_available = find_available_gpus()
    num_gpus = len(gpu_available)

    global_num_gpu_threads = {device_name: 0 for device_name in gpu_available}


    def train_rl_agent_trial(trial: optuna.Trial):
        #query which gpu to occupy
        device = ""
        global_lock.acquire()
        for k, v in global_num_gpu_threads.items():
            if v < num_threads_per_gpu:
                device = k
                break
        global_num_gpu_threads[device] += 1
        global_lock.release()

        if not device:
            raise RuntimeError("No available GPU devices.")

        try:
            current_logdir = path.join(logdir, "{}".format(trial.number))

            #make log dir
            os.makedirs(current_logdir, exist_ok=True)

            #suggest hyperparameters
            hyperparameters = {
                "lr_actor":       trial.suggest_loguniform("lr_actor",     1e-4, 1e-3),
                "lr_critic":      trial.suggest_loguniform("lr_critic",    1e-4, 1e-3),
                "repeat":         5 * trial.suggest_int("repeat", 1, 8),
                "target_entropy": trial.suggest_uniform("target_entropy", -5, 0),
            }

            #update env config
            hyperparameters.update({
                "env_name": env_name,
                "env_reward_threshold": env_reward_threshold
            })

            #update config
            hyperparameters.update({
                "max_env_steps": max_env_steps,
                "parallel_envs": parallel_envs,
                "log_dir": current_logdir,
                "device": device
            })

            #write hyperparameter set
            with open(path.join(current_logdir, "hyperparameters.json"), "w") as f:
                json.dump(hyperparameters, f, indent=4)
                f.close()

            #train in subprocess
            pipe_parent, pipe_child = Pipe()
            train_process = Process(target=train_rl_agent_worker, args=(pipe_child, hyperparameters))
            train_process.start()

            try:
                while True:
                    msg, payload = pipe_parent.recv()
                    if msg == "shouldprune":
                        trial.report(payload["rew"], payload["step"])
                        pipe_parent.send(trial.should_prune())
                    elif msg == "pruned":
                        raise optuna.TrialPruned()
                    elif msg == "error":
                        raise payload
                    elif msg == "end":
                        return payload
            except KeyboardInterrupt:
                train_process.kill()
            finally:
                train_process.join()

        finally:
            #release occupied gpu
            global_lock.acquire()
            global_num_gpu_threads[device] -= 1
            global_lock.release()

        return None

    #make log dir
    os.makedirs(logdir, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=int(min_trial_env_steps),
            max_resource=int(max_env_steps),
            reduction_factor=reduction_factor
        ),
        #SQLite do not support multithreading!!!
        #storage=os.path.join("sqlite:///", logdir, "optuna.db"),
        #load_if_exists=True
    )

    #print brackets
    brackets = np.floor(np.log(max_env_steps / min_trial_env_steps) / np.log(reduction_factor)) + 1
    if brackets < 4 or brackets > 6:
        print("[WARN] Bracket number should be in [4,6].")

    #print info
    print("Parallel envs: {}\nSearch jobs: {}\nHyperband brackets: {}".format(parallel_envs, num_gpus * num_threads_per_gpu, brackets))

    study.optimize(train_rl_agent_trial,
        n_trials=num_trials,
        n_jobs=num_gpus * num_threads_per_gpu
    )
    print(study.best_params)

if __name__ == '__main__':
    train_rl_agent_auto()