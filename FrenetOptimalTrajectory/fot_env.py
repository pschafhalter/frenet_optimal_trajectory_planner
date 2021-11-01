import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib
import time
from multiprocessing import Process

import fot_planner
import fot_wrapper
from fot_wrapper import (
    _parse_hyperparameters,
    to_frenet_initial_conditions,
    query_anytime_planner_path,
)

matplotlib.use("TkAgg")


def generate_obstacles(num_obstacles: int):
    # TODO: smarter generation of obstacles.
    obstacles = []
    for _ in range(num_obstacles):
        llx = np.random.randint(0, 100)
        lly = np.random.randint(-5, 2)
        urx = llx + np.random.randint(0, 6)
        ury = lly + np.random.randint(0, 6)
        obstacles.append([llx, lly, urx, ury])
    return np.array(obstacles)


def generate_initial_conditions(num_obstacles: int):
    return {
        "ps": 0,
        "target_speed": np.random.randint(15, 26),
        "pos": np.array([0, 0]),
        "vel": np.array([np.random.randint(0, 26), 0]),
        "wp": np.array([[0, 0], [50, 0], [150, 0]]),
        "obs": generate_obstacles(num_obstacles),
    }


class FOTEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Dict(
            {
                "d_road_w": spaces.Box(1e-3, 5.0, shape=(1,)),
                "dt": spaces.Box(1e-3, 1.0, shape=(1,)),
                "mint": spaces.Box(0.1, 10.0, shape=(1,)),
                "d_min_max_t": spaces.Box(0.1, 10.0, shape=(1,)),
                "d_t_s": spaces.Box(1e-3, 5.0, shape=(1,)),
                "n_s_sample": spaces.Discrete(10),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "num_obstacles": spaces.Discrete(11),
                "deadline": spaces.Box(0.001, 0.1, shape=(1,)),
            }
        )

        self.initial_conditions = None

    def reset(self):
        self.num_obstacles = np.random.randint(0, 11)
        self.initial_conditions = generate_initial_conditions(self.num_obstacles)
        self.deadline = np.random.uniform(0.001, 0.1)
        return {
            "num_obstacles": self.num_obstacles,
            "deadline": np.array([self.deadline], dtype=np.float32),
        }

    def render(self):
        wx = self.initial_conditions["wp"][:, 0]
        wy = self.initial_conditions["wp"][:, 1]
        obs = np.array(self.initial_conditions["obs"])

        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        plt.plot(wx, wy)
        if obs.shape[0] == 0:
            obs = np.empty((0, 4))
        ax = plt.gca()
        for o in obs:
            rect = patch.Rectangle((o[0], o[1]), o[2] - o[0], o[3] - o[1])
            ax.add_patch(rect)
        plt.plot(self.result_x[1:], self.result_y[1:], "-or")
        plt.plot(self.result_x[1], self.result_y[1], "vc")
        plt.xlim(0, 110)
        plt.ylim(-6, 6)
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.title("v[m/s]:" + str(np.linalg.norm(self.initial_conditions["vel"]))[0:4])
        plt.grid(True)
        plt.pause(0.5)
        # plt.pause(1)

    def step(self, action: dict):
        hyperparameters = {
            "max_speed": 25.0,
            "max_accel": 15.0,
            "max_curvature": 15.0,
            "max_road_width_l": 5.0,
            "max_road_width_r": 5.0,
            "d_road_w": None,  # 0.5,
            "dt": None,  # 0.2,
            "maxt": None,  # 5.0,
            "mint": None,  # 2.0,
            "d_t_s": None,  # 0.5,
            "n_s_sample": None,  # 2.0,
            "obstacle_clearance": 0.1,
            "kd": 1.0,
            "kv": 0.1,
            "ka": 0.1,
            "kj": 0.1,
            "kt": 0.1,
            "ko": 0.1,
            "klat": 1.0,
            "klon": 1.0,
            "num_threads": 0,  # set 0 to avoid using threaded algorithm
        }

        action["maxt"] = action["mint"] + action.pop("d_min_max_t")
        action["n_s_sample"] += 1
        hyperparameters.update(action)
        frenet_hp = _parse_hyperparameters(hyperparameters)
        frenet_ic, _ = to_frenet_initial_conditions(self.initial_conditions)

        planner = fot_planner.FotPlanner(frenet_ic, frenet_hp)

        # TODO: plan for a certain amount of time.
        planner.async_plan()
        start = time.time()
        while time.time() < start + self.deadline:
            pass

        (
            self.result_x,
            self.result_y,
            speeds,
            ix,
            iy,
            iyaw,
            d,
            s,
            speeds_x,
            speeds_y,
            misc,
            costs,
            success,
            best_fot_rv_so_far,
        ) = query_anytime_planner_path(planner, return_rv_object=True)

        # May need to move this down.
        planner.stop_plan()

        if success:
            self.initial_conditions["pos"] = np.array(
                [self.result_x[1], self.result_y[1]]
            )
            self.initial_conditions["vel"] = np.array([speeds_x[1], speeds_y[1]])
            self.initial_conditions["ps"] = misc["s"]
            done = self.result_x[1] > 100
            reward = costs["cf"]
        else:
            done = True
            reward = -1000

        # self.deadline = 0.2
        self.deadline = np.random.uniform(0.001, 0.1)

        obs = {
            "num_obstacles": self.num_obstacles,
            "deadline": np.array([self.deadline]),
        }
        return obs, reward, done, {}


# env = FOTEnv()
# print(env.observation_space.sample())

if __name__ == "__main__":
    env = FOTEnv()
    env.reset()

    for _ in range(1000):
        obs, reward, done, info = env.step(env.action_space.sample())
        # env.render()
        print(f"reward: {reward}")
        if done:
            env.reset()
