# Skydio X2 in Mujoco

## Getting started
Set-up MuJoCo and the requirements. Then, simply run main.py

## To-Dos

- Improve PID controllers
- Add sensor and state estimator
- Add trajectory planner utilizing PPO algorithm 

## PPO acrobatic courses

`main.py` now contains a PPO training loop around the `DroneAcroEnv`, which samples
different acrobatic loop courses (vertical loop, banked loop, figure-eight, and
corkscrew) every episode. Each course comes with a unique arrangement of
spherical obstacles that the quadrotor must clear. The observation space exposes
the drone state along with the current desired position, velocity, and normalized
progress through the active course. Rewards encourage tight trajectory tracking,
positive course progress, and safe clearance from obstacles while heavily
penalizing collisions or leaving the safe flight envelope.

### Usage

1. Install MuJoCo dependencies plus `stable-baselines3`.
2. Adjust `MODE = "train"` in `main.py` to learn a policy:
   - Training outputs TensorBoard data under `ppo_drone_tb/` and a policy file
     `ppo_drone_loop.zip`.
3. Switch `MODE` back to `"play"` to watch the learned agent fly through the
   randomly sampled courses in the MuJoCo viewer (obstacle/course info is shown
   on the console).

You can also import `DroneAcroEnv` from `main.py` in your own scripts to plug it
into different algorithms or to iterate on reward design.

### Available courses

- `vertical_loop` – power loop in the x-z plane with staggered pillars.
- `banked_horizontal_loop` – fast horizontal circle threaded through hoops.
- `figure_eight` – classic figure-eight with a tight crossover ring.
- `corkscrew` – tall spiral with suspended spheres to dodge.

## Acknowledgements
Mujoco: https://github.com/google-deepmind/mujoco
Mujoco Menagerie: https://github.com/google-deepmind/mujoco_menagerie
Simple PID: https://github.com/m-lundberg/simple-pid/tree/master
