1. Gym 安装

   ```shell
   pip install gym
   
   # 安装各种环境
   git clone https://github.com/openai/gym.git
   cd gym
   # pip install -e .    安装 gym
   pip install -e '.[atari]'
   pip install -e '.[board_game]'
   pip install -e '.[box2d]'
   pip install -e '.[classical_control]'
   pip install -e '.[mujoco]'
   ```



2. base

   ```python
   # 查看所有可用的 env:   gym.envs.registry.env_specs
   import gym
   env = gym.make('[game name]')
   env.reset()
   observation, reward, done, info = env.step(env.action_space.sample())
   env.render()
   
   env.action_space
   env.observation_space	.high	.low
   ```



3. spaces

   ```python
   space.sample()	# 采样一个动作
   space.n		# 空间大小
   
   # Discrete gym.spaces.Discrete
   Discrete(n)	# 定义一个离散空间
   
   
   # Box gym.spaces.Box
   Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
   Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
   
   ```




 4. envs

    ```
    # Toy text
    Blackjask-v0	FrozenLake-v0	FrozenLake8X8-v0	GuessingGame-v0
    HotterColder-v0	NChain-v0		Roulette-v0			Taxi-v2
    
    # Algorithms
    Copy-v0
    DuplicatedInput-v0
    RepeatCopy-v0
    Reverse-v0
    ReversedAddition-v0
    ReversedAddition3-v0
    ```

    