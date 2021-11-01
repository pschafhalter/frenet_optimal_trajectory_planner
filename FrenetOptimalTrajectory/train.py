from fot_env import FOTEnv

import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ddpg.ddpg import DEFAULT_CONFIG
import ray.rllib.agents.ddpg as ddpg


def env_creator(env_config):
    return FOTEnv()  # return an env instance


register_env("FOTEnv", env_creator)

config = {
    "env": "FOTEnv",
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    # Max global norm for each gradient calculated by worker
    "grad_clip": 40.0,
    # Workers sample async. Note that this increases the effective
    # rollout_fragment_length by up to 5x due to async buffering of batches.
    "num_workers": 32,
    "num_gpus": 1,
    # "num_gpus_per_worker": 0.5, # (2 - 0.2) / 2,
    "framework": "torch",
    "actor_hiddens": [16, 16],
    "actor_hidden_activation": "relu",
    "critic_hiddens": [16, 16],
    "critic_hidden_activation": "relu",
    "n_step": 1,
    "evaluation_interval": 5,
    "evaluation_num_episodes": 10,
    "critic_lr": 1e-2,
    # Learning rate for the actor (policy) optimizer.
    "actor_lr": 1e-2,
    "train_batch_size": 512,
    "target_network_update_freq": 500000,
     "timesteps_per_iteration": 25000,
     # "exploration_config": {"type": "GaussianNoise"},
}

ray.shutdown()
ray.init()

trainer = ddpg.ApexDDPGTrainer(config=config)
# trainer.restore("/home/eecs/pschafhalter/ray_results/APEX_DDPG_FOTEnv_2021-11-01_12-22-55rerdzao1/checkpoint_000011/checkpoint-11")

for i in range(100000):
    print(f"Iter: {i}")
    # print(trainer.train())
    trainer.train()
    trainer.save()
