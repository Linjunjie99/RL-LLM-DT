# olympics_engine

Here lies the underlying physics engine for the curling environment, including the collision detection and response, map generation and pygame visualisation.

To play a game, run the *main.py* file or execute:

> python main.py

## Partial observation

In every map, agents can only see object in front of him, this includes the bouncable wall, crossable line and opponent agents.

The observation is a 30*30 array shown as below:

<img src=https://github.com/jidiai/Competition_Olympics-Curling/blob/main/olympics_engine/assets/agent_view.png>

