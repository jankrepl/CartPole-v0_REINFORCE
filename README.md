# CartPole_v0 REINFORCE algorithm
Solution to the CartPole_v0 environment using the general **REINFORCE** algoritm.

## Code

### Running
```
python Main.py
```

### Dependencies
*  gym
*  numpy
*  tensorflow

## Detailed Description
### Problem Statement and Environment
The goal is to move the cart to the left and right in a way that the pole on top of it does not fall down. The states 
of the environment are composed of 4 elements - **cart position** (x), **cart speed** (xdot),
**pole angle** (theta) and **pole angular velocity** (thetadot). For each time step when the pole is still on the cart
we get a reward of 1. The problem is considered to be solved if for 100 consecutive
episodes the average reward is at least 195.


If we translate this problem into reinforcement learning terminology:
* action space is **0** (left) and **1** (right)
* state space is a set of all 4-element lists with all possible combinations of values of x, xdot, theta, thetadot

---
### REINFORCE algorithm
We are going to look for the optimal strategy without using any value functions - **actor only approach**.
We restrict ourselves to finding only deterministic policies and parametrize them in a following way:

```
argmax(softmax(sW))

where

s.shape = (1,4)
W.shape = (4,2)
```

To update weights we use the Policy Gradient Theorem and from it derived algorithm REINFORCE:

![screen shot 2017-09-12 at 4 30 13 pm](https://user-images.githubusercontent.com/18519371/30331165-4ee23f00-97d7-11e7-8774-aa7ea42e6e03.png)

Gradient of the log policy can be derived both analytically or computed by TensorFlow - we implemented both.

## Results and discussion
This method has a very high variance and also performs differently for different initializations of weights.
See below one evolution of the algorithm performance over 5000 episodes (the maximum score is 200)

![screen shot 2017-09-12 at 4 23 20 pm](https://user-images.githubusercontent.com/18519371/30331348-dab2cc2a-97d7-11e7-9b0e-1bb6dccaf209.png)


## Resources and links
* ![Original Paper](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
* ![David Silver - Policy Gradient Lecture Slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
