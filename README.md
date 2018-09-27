This is a repo that contains code demonstrating the **work of DQN agent that searches known-value optimum** over the field of 20x20 steps  
Enviroment allows agent to move in up-down-left-right directions over the **discrete set of coords on the 2D field** of 20x20 points  
Agent knows the optimum value it needs to find and **learns to stably reach it** from any point in a final time (like **12-14k iterations**)
Methods used:
* step penalty
* edge-touching penalty
* reward for optimum
* path non-consistency penalty
* feeding last 3 steps coords
* feeding last 3 steps fit functions
* 96-64-64-32 relu fully connected
* BoltzmannGumbelQPolicy

Code runs with rendering and supports gym interface.
