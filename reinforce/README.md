## Reinforce(Monte-Carlo Policy Gradient)

![reinforce](../image/reinforce.png)



- run

  ```
  python main.py
  
  # line: 23~24
  # istrain: train or eval
  # perfomance_render: when eval, show the environment
  ```



- loss

  [reference](https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/3-reinforce/cartpole_reinforce.py) 

  categorical cross entropy:
  $$
  H(p, q) = \sum{p_i*log(q_i)}
  $$
  take action - $a$ :
  $$
  p_a = value \\
  q_a = policy(s, a) \\
  $$
  then we get
  $$
  J_{max} = reward * log(\pi_a)
  $$
  which mean we should maximize the action what we have selected .

​		