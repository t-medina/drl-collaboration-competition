[//]: # (Image References)

[rewards_plot]: rewards-plot.png "Rewards Plot"
[trained_agent]: trained.gif "Trained Agent"

# Project report

## Learning algorithm

The learning algorithm used is DDPG. Upon a first read, I decided to attempt the same DDPG implementation I did for the [Continous Control](https://github.com/t-medina/drl-continuous-control) problem. That agent performed quite well for that environment, and I thought I could start from there and build on a solution for this problem.

I left the agent and model unmodified, and changed slightly the way the agent is used. On the Tennis environment we have two agents, and both receives its own, local observation. The rewards are computed in the same way, and the observation and actions, though relative to each agent, also belong to the same state and action space. So I thought I could use the Agent in the exact same way I used it in the **Continous Control**, except that in every timestep I would have two experiences to feed the Agent and learn from. I really liked from this idea that I could use both agents experiences, and that as the Agent kept learning and improving, better and better actions would be taken by both agents.

The steps were:
- given the state observation of agent1, choose an action1
- given the state observation of agent2, choose an action2
- pass both action1 and action2 to the environment
- use the action1, the previous and new state observation of agent1 and the reward obtained by agent1 to train the Agent
- use the action2, the previous and new state observation of agent2 and the reward obtained by agent2 to train the Agent

```python
    action1 = agent.act(states[0])                 # choose best action, given agent1 state
    action2 = agent.act(states[1])                 # choose best action, given agent2 state
    
    actions = [action1, action2]
    env_info = env.step(actions)[brain_name]       # send the actions to the environment
    
    next_states = env_info.vector_observations     # get the next state (for each agent)
    rewards = env_info.rewards                     # get the reward (for each agent)
    dones = env_info.local_done                    # see if episode has finished(for any of the agent)
    
    # use each agent reward and new state, as well as the previous state and action taken to train the agent
    agent.step(states[0], action1, rewards[0], next_states[0], dones[0])
    agent.step(states[1], action2, rewards[1], next_states[1], dones[1])
```

My intention was to give this idea a try and build from it. But it worked wonderfully on the first attempt! The learning started out slowly, but the environment was resolved in just **586 episodes**. Excellent results were achieved later in the testing phase, where a consistent score of over **2.6** was obtained.

The algorithm implments an actor-critic method, so two neural network were implemented. 

The actor neural network consists of three layers:

- Fully connected layer - input: 24, output: 128
- Fully connected layer - input: 128, output: 128
- Fully connected layer - input: 128, output: 2

The input and output correspond to the state space size (24) and action space size (2)

The crtic neural network also consists of three layers, with an slightly different structure:

- Fully connected layer - input: 24, output: 128
- Fully connected layer - input: 128 + 2, output: 128
- Fully connected layer - input: 128, output: 1

Batch normalization was added after the first layer of both the actor and the critic.

### Hyperparameters
Parameter | Value
--- | ---
replay buffer size | 100000
batch size | 128
gamma | 0.99
tau | 0.001
actor learning rate | 0.0002
critic learning rate | 0.0002
num episodes | 1000
weight decay | 0

## Results

### Plot of Rewards

The environment was solved in 210 episodes:

![Rewards Plot][rewards_plot]

```
Episode 25	Average Score: 0.00
Episode 50	Average Score: 0.00
Episode 75	Average Score: 0.00
Episode 100	Average Score: 0.00
Episode 125	Average Score: 0.01
Episode 150	Average Score: 0.01
Episode 175	Average Score: 0.01
Episode 200	Average Score: 0.01
Episode 225	Average Score: 0.00
Episode 250	Average Score: 0.00
Episode 275	Average Score: 0.00
Episode 300	Average Score: 0.00
Episode 325	Average Score: 0.01
Episode 350	Average Score: 0.03
Episode 375	Average Score: 0.05
Episode 400	Average Score: 0.07
Episode 425	Average Score: 0.09
Episode 450	Average Score: 0.10
Episode 475	Average Score: 0.11
Episode 500	Average Score: 0.13
Episode 525	Average Score: 0.13
Episode 550	Average Score: 0.22
Episode 575	Average Score: 0.34
Episode 586	Average Score: 0.52
Environment solved in 586 episodes!	Average Score: 0.52
```

### Trained agent

![Trained Agent][trained_agent]

## Ideas for future work

I would like to try algorithms like PPO, A3C and D4PG.