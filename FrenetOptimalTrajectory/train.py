from fot_env import FOTEnv

from ray.tune.registry import register_env
from ray.rllib.agents.a3c.a3c import DEFAULT_CONFIG
import ray.rllib.agents.a3c as a3c


def env_creator(env_config):
    return FOTEnv()  # return an env instance


register_env("FOTEnv", env_creator)

config = {
    "env": "FOTEnv",
    "use_critic": True,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": False,
    # Size of rollout batch
    "rollout_fragment_length": 100,
    # Max global norm for each gradient calculated by worker
    "grad_clip": 40.0,
    # Learning rate
    "lr": 0.01,
    # Learning rate schedule
    "lr_schedule": None,
    # Value Function Loss coefficient
    "vf_loss_coeff": 0.5,
    # Entropy coefficient
    "entropy_coeff": 0.01,
    # Min time per iteration
    "min_iter_time_s": 5,
    # Workers sample async. Note that this increases the effective
    # rollout_fragment_length by up to 5x due to async buffering of batches.
    "sample_async": True,
    # "rollout_fragment_length": 100,
    # "train_batch_size": 100,
    "num_workers": 1,
}

trainer = a3c.A3CTrainer(config=config)
for i in range(10):
    print(f"Iter: {i}")
    print(trainer.train())
    print(trainer.save())
