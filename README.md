## Modified MineDojo Package

The code is a slightly modified version of [MineDojo](https://github.com/MineDojo/MineDojo) used for [COPL](https://github.com/PKU-RL/COPL). 

### Installation

Please refer to this [document](https://docs.minedojo.org/sections/getting_started/install.html) for the package installation and use. Note that `Python>=3.9` is required.

Install modified MineDojo:
```shell
git clone https://github.com/PKU-RL/MCEnv_COPL.git
cd MCEnv_COPL
pip install -e .
```

### Modifications on MineDojo

* We support multi-task environment setting, which is able to include combat tasks and harvest tasks simutaneously. See `minedojo/tasks/meta/multi_task.py`.
* We support randomly teleporting the player in the fast_reset mode: when calling `env.reset()`, the player will be teleported to a random location near its initial spawn point.

