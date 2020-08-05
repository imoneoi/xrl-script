"""
    Implement your agent training code here
"""

def train_rl_agent(
    #These arguments are required for AutoRL

    #Pruner
    prune_func: Optional[Callable[[int, float], bool]] = None,

    #Selected environment
    env_name: str = "LunarLanderContinuous-v2",
    env_reward_threshold: Optional[int] = 200,

    #Train hyperparameters
    max_env_steps: int = 1e7,

    #Number of parallel vector environments
    parallel_envs: int = 8,

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",

    # Save dir
    log_dir: str = "logs/",

    #Define your algorithm hyperparameters here
    # buffer_size: int = 1000000,
    # lr_actor: float = 3e-4,
    # lr_critic: float = 1e-3,

    # alpha: Union[float, str] = "auto",
    # target_entropy: Union[float, str] = "auto",

    # gamma: float = 0.99,
    # tau: float = 0.005,
) -> None:
    #Train Pruner, Average last 5 tests
    test_rew_movavg = MovAvg(5)
    def TrainPruner(env_step, test_reward):
        test_rew_movavg.push(test_reward)

        #Pruning
        if prune_func:
            prune_func(test_rew_movavg.get(), env_step)

    #Implement your trainging logic here
    #Remember to call TrainPruner every testing step
    #...

    return test_rew_movavg.get()