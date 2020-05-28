[//]: # (Image References)

[trained_agent]: trained.gif "Trained Agent"
[untrained_agent]: untrained.gif "Untrained Agent"


# Project 3: Collaboration and Competition

### Introduction

This project implements a DDPG agent capable of playing tennis.


Untrained Agent             |  Trained Agent
:-------------------------:|:-------------------------:
![Untrained Agent][untrained_agent] | ![Trained Agent][trained_agent]


For this project, the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment was used.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of **+0.1**.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of **-0.01**.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of **+0.5** (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least **+0.5**.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the same folder as the `Tennis.ipynb` file, and unzip (or decompress) the file. 

### Instructions

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drl python=3.6
	conda activate drl
	```
	- __Windows__: 
	```bash
	conda create --name drl python=3.6 
	activate drl
	```
	
2. Navigate to the location where you cloned this repository, and install the python dependencies.
    ```bash
    pip install .
    ```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drl` environment.  
    ```bash
    python -m ipykernel install --user --name drl --display-name "drl"
    ```

4. Before running code in a notebook, change the kernel to match the `drl` environment by using the drop-down `Kernel` menu. 

5. Follow the instructions in `Tennis.ipynb`:
    - Cells in sections 1, 2 and 3 will allow you to initialize and explore the environment
    - Cells in section 4 include the training of the agent
    - Cells in section 5 include the testing of the trained agent. Saved model weights of a successfully trained agent are included in this repository as well (both actor and critic: `actor_weights.pth` and `critic_weights.pth`), so you can skip training and go straight to this section to see a trained agent perform.