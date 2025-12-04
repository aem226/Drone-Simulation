import time
import os
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import mujoco
import mujoco.viewer

import gymnasium as gym
from gymnasium import spaces

# Utility functions

def quat_to_euler_xyz(q):
    """Convert quaternion [w,x,y,z] -> roll,pitch,yaw in XYZ order."""
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return roll, pitch, yaw


def figure8_trajectory(t):
    # Horizontal figure-8 centered around y ≈ 6.5, z ≈ 2
    A = 2.0   # left-right span in x
    B = 1.5   # extent in y around center
    y0 = 6.5
    z0 = 2.0
    speed = 0.5

    theta = speed * t

    x = A * np.sin(theta)
    y = y0 + B * np.sin(theta) * np.cos(theta)
    z = z0

    return np.array([x, y, z])


# Course / Obstacle / Specs

@dataclass
class Obstacle:
    position: Sequence[float]
    radius: float
    penalty: float = 100.0
    name: str = ""

    def __post_init__(self):
        self.position = np.asarray(self.position)

    def clearance(self, point):
        return float(np.linalg.norm(point - self.position) - self.radius)


@dataclass
class CourseSpec:
    name: str
    trajectory_fn: Callable[[float], np.ndarray]
    duration: float
    description: str
    completion_bonus: float = 150.0
    obstacles: Sequence[Obstacle] = ()

    def __post_init__(self):
        self.obstacles = tuple(self.obstacles)

    def desired_state(self, t):
        tau = np.clip(t, 0, self.duration)
        pos = self.trajectory_fn(tau)
        eps = 1e-3
        pos2 = self.trajectory_fn(np.clip(tau + eps, 0, self.duration))
        vel = (pos2 - pos) / eps
        return pos, vel


# Drone low-level MuJoCo wrapper

class Drone:
    def __init__(self, xml_path="mujoco_menagerie-main/skydio_x2/scene.xml"):
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)

        ctrl_range = self.m.actuator_ctrlrange[:4]
        self.u_center = 0.5 * (ctrl_range[:, 0] + ctrl_range[:, 1])
        self.u_half = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:3] = np.array([0, 0, 1.5])
        self.d.qpos[3:7] = np.array([1, 0, 0, 0])
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)
        return self.get_obs()

    def get_obs(self):
        return np.concatenate([self.d.qpos.copy(), self.d.qvel.copy()])

    def step_with_action(self, action):
        action = np.clip(action, -1, 1)
        u = self.u_center + self.u_half * action
        self.d.ctrl[:4] = u
        mujoco.mj_step(self.m, self.d)

    def get_pos_vel_att(self):
        pos = self.d.qpos[:3].copy()
        vel = self.d.qvel[:3].copy()
        quat = self.d.qpos[3:7]
        phi, theta, psi = quat_to_euler_xyz(quat)
        return pos, vel, np.array([phi, theta, psi])


# Environment with RL logic 

class DroneAcroEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, max_time=12.0, xml_path="mujoco_menagerie-main/skydio_x2/scene.xml"):
        super().__init__()

        self.drone = Drone(xml_path)
        self.hover_time = 0.5
        self.max_time = max_time

        self.course = CourseSpec(
            name="figure8_hoops",
            trajectory_fn=figure8_trajectory,
            duration=10.0,
            description="Horizontal figure-8 through four hoops",
            obstacles=(
                Obstacle(position=[-2.0, 5.0, 2.0], radius=0.6, penalty=120.0, name="hoop1"),
                Obstacle(position=[ 2.0, 5.0, 2.0], radius=0.6, penalty=120.0, name="hoop2"),
                Obstacle(position=[-2.0, 8.0, 2.0], radius=0.6, penalty=120.0, name="hoop3"),
                Obstacle(position=[ 2.0, 8.0, 2.0], radius=0.6, penalty=120.0, name="hoop4"),
            ),
        )

        nq = self.drone.m.nq
        nv = self.drone.m.nv
        self.obs_dim = nq + nv + 7
        self.action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32)

        self.t0 = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t0 = 0.0

        obs = self.drone.reset()
        target_pos, target_vel, progress = self._desired_state(0.0)
        full_obs = self._compose_obs(obs, target_pos, target_vel, progress)
        return full_obs.astype(np.float32), {}

    def _desired_state(self, t):
        if t < self.hover_time:
            return np.array([0, 0, 1.5]), np.zeros(3), 0.0
        ct = t - self.hover_time
        pos, vel = self.course.desired_state(ct)
        progress = ct / self.course.duration
        return pos, vel, progress

    def _compose_obs(self, base, tpos, tvel, prog):
        return np.concatenate([base, tpos, tvel, np.array([prog])])

    def step(self, action):
        self.drone.step_with_action(action)
        t = self.drone.d.time

        pos, vel, angles = self.drone.get_pos_vel_att()
        target_pos, target_vel, progress = self._desired_state(t)

        pos_err = pos - target_pos
        vel_err = vel - target_vel

        reward = -(pos_err @ pos_err + 0.1 * (vel_err @ vel_err))

        # Hoop rewards / penalties
        for obs in self.course.obstacles:
            dist = np.linalg.norm(pos - obs.position)
            if dist < 0.3:                 # very close to hoop center
                reward += 60.0
            elif dist < obs.radius:        # inside the ring
                reward += 25.0
            elif dist < obs.radius + 0.3:  # near the frame: treat like near-crash
                reward -= obs.penalty

        phi, theta, psi = angles
        reward -= 0.3 * (phi ** 2 + theta ** 2)

        terminated = False
        truncated = False

        if pos[2] < 0.05:
            reward -= 20.0

        if t - self.hover_time > self.course.duration:
            reward += self.course.completion_bonus
            terminated = True

        if np.linalg.norm(pos) > 25.0:
            reward -= 100.0
            terminated = True

        if t > self.max_time:
            truncated = True

        full_obs = self._compose_obs(self.drone.get_obs(), target_pos, target_vel, progress)
        info = dict(t=t, pos=pos, vel=vel, progress=progress)
        return full_obs.astype(np.float32), reward, terminated, truncated, info


# PPO + training / playback

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return DroneAcroEnv()

def train_ppo(total_timesteps=300_000, model_path="ppo_drone_figure8.zip"):
    env = DummyVecEnv([make_env])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=256,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./ppo_drone_tensorboard",
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    env.close()

def play_ppo(model_path="ppo_drone_figure8.zip"):
    if not os.path.exists(model_path):
        print("Model not found:", model_path)
        return
    env = DroneAcroEnv()
    model = PPO.load(model_path)
    drone = env.drone
    with mujoco.viewer.launch_passive(drone.m, drone.d) as viewer:
        obs, _ = env.reset()
        done = False
        truncated = False
        step_count = 0
        episode = 0
        while viewer.is_running():
            start = time.time()
            if done or truncated:
                print(f"[INFO] Episode {episode} done after {step_count} steps")
                obs, _ = env.reset()
                episode += 1
                step_count = 0
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            viewer.sync()
            dt = drone.m.opt.timestep - (time.time() - start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    MODE = "play"  # change to "play" after training once

    if MODE == "train":
        train_ppo(300_000)
    else:
        play_ppo()
