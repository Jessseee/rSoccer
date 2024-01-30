# RSoccer SSL and VSSS Gym environments
RSoccer Gym is an open-source framework to study Reinforcement Learning (RL) for SSL and IEEE VSSS competition environment. The simulation is done by [rSim](https://github.com/robocin/rsim). This fork adds a number of RL agent implementations from the [CleanRL](https://github.com/vwxyzjn/cleanrl) library to start training on the environments.

## Reference
If you use this environment in your publication and want to cite the original authors, utilize this BibTeX:

```
@InProceedings{10.1007/978-3-030-98682-7_14,
    author          = {Martins, Felipe B.
                       and Machado, Mateus G.
                       and Bassani, Hansenclever F.
                       and Braga, Pedro H. M.
                       and Barros, Edna S.},
    editor          = {Alami, Rachid
                       and Biswas, Joydeep
                       and Cakmak, Maya
                       and Obst, Oliver},
    title           = {rSoccer: A Framework for Studying Reinforcement 
                       Learning in Small and Very Small Size Robot Soccer},
    booktitle       = {RoboCup 2021: Robot World Cup XXIV},
    year            = {2022},
    publisher       = {Springer International Publishing},
    address         = {Cham},
    pages           = {165--176},
    isbn            = {978-3-030-98682-7}
}
```

## Install from source
```bash
git clone https://github.com/Jessseee/rSoccer.git
cd rSoccer
pip install .
```
For [editable installs](https://setuptools.pypa.io/en/latest/userguide/development_mode.html), change last command to `"pip install -e ."`.
# Available Envs

|           IEEE VSSS            |
|:------------------------------:|
| ![](.github/resources/vss.gif) |

|         Static Defenders          |              Contested Possession               |
|:---------------------------------:|:-----------------------------------------------:|
| ![](.github/resources/static.gif) | ![](.github/resources/contested_possession.gif) |

|              Dribbling               |              Pass Endurance               |
|:------------------------------------:|:-----------------------------------------:|
| ![](.github/resources/dribbling.gif) | ![](.github/resources/pass_endurance.gif) |

|                                  Environment Id                                  | Observation Space | Action Space | Step limit |
|:--------------------------------------------------------------------------------:|:-----------------:|:------------:|:----------:|
|                    [VSS-v0](rsoccer_gym/vss/README.md#vss-v0)                    |     Box(40,)      |   Box(2,)    |    1200    |
|     [SSLStaticDefenders-v0](rsoccer_gym/ssl/README.md#sslstaticdefenders-v0)     |     Box(24,)      |   Box(5,)    |    1000    |
|           [SSLDribbling-v0](rsoccer_gym/ssl/README.md#ssldribbling-v0)           |     Box(21,)      |   Box(4,)    |    4800    |
| [SSLContestedPossession-v0](rsoccer_gym/ssl/README.md#sslcontestedpossession-v0) |     Box(14,)      |   Box(5,)    |    1200    |
|       [SSLPassEndurance-v0](rsoccer_gym/ssl/README.md#sslpassendurance-v0)       |     Box(18,)      |   Box(3,)    |    1200    |


# Example code

## Environment

```Python
import numpy as np
from gymnasium.spaces import Box
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv


class SSLExampleEnv(SSLBaseEnv):
    def __init__(self, render_mode=None):
        super().__init__(
            field_type=0,  # SSL Division A Field
            n_robots_blue=1,
            n_robots_yellow=0, 
            time_step=0.025,
            render_mode=render_mode
        )
        self.action_space = Box(
            low=-1,
            high=1,
            shape=(2, )  # Robot v_x, v_y
        )
        self.observation_space = Box(
            low=-self.field.length / 2,
            high=self.field.length / 2,
            shape=(4, )  # Ball x, y and Robot x, y
        )

    def _frame_to_observations(self):
        ball, robot = self.frame.ball, self.frame.robots_blue[0]
        return np.array([ball.x, ball.y, robot.x, robot.y])

    def _get_commands(self, actions):
        return [Robot(
            yellow=False, 
            id=0,
            v_x=actions[0], 
            v_y=actions[1]
        )]

    def _calculate_reward_and_done(self):
        half_length = self.field.length / 2
        half_width = self.field.goal_width / 2
        ball_in_goal = self.frame.ball.x > half_length and abs(self.frame.ball.y) < half_width
        if ball_in_goal:
            reward, done = 1, True
        else:
            reward, done = 0, False
        return reward, done
    
    def _get_initial_positions_frame(self):
        pos_frame: Frame = Frame()
        pos_frame.ball = Ball(
            x=(self.field.length/2) - self.field.penalty_length, 
            y=0.
        )
        pos_frame.robots_blue[0] = Robot(x=0., y=0., theta=0,)
        return pos_frame

```

## Custom Agent

```Python
import gymnasium as gym
import rsoccer_gym  # This registers the environments

# Using VSS Single Agent env
env = gym.make('VSS-v0', render_mode="human")

env.reset()
# Run for 1 episode and print reward at the end, this would be your training loop
for i in range(1):
    terminated = False
    truncated = False
    while not (terminated or truncated):
        # Step using random actions
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
    print(reward)

```
