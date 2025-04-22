# Maze Games 

Maze games, also known as grid world games, are the most popular test environments for QRL algorithms. However, in QRL literature, various researchers have used different maze games, which makes it difficult to compare the results. In this tutorial, we will show how to implement a custom maze game env which can be used with CleanQRL. We proivde an implementation which makes it easy to customize parameters like rewards and maze size such that one can adapt it easily to implementations provided in the literature. 

Many researchers have used different maze games with sligth differences:

| Researcher/Paper       |  Maze Size        | Reward Structure (reward, penalty, neutral)      | maze name    |
|------------------|-------------------|------------------------|-----------------|
| [Crawford](https://arxiv.org/pdf/1612.05695)   | nx5         | 200, 0, 100     | crawford            |
| [Müller](https://arxiv.org/pdf/2109.10900)     | 3x3         | 220, -220, -10  | mueller |
| [Neumann](https://link.springer.com/content/pdf/10.1007/s11128-023-03867-9.pdf)    | 3x4       | 200, -200, -10        | neumann_a            |
|                                                                                    | 3x5     | 200, -200, -10        |  neumann_b           |
|                                                                                    | 4x5  | 200, -200, -10        |  neumann_c          |

Depending on the QRL algorithm, the state is either binary or one-hot encoded. Additionally, sometimes the actions of the agents are restricted and sometimes they are not. As one can see, this makes meaningful comparisons difficult. Nevertheless, in the implementation below, one can customize the following parameters:


```py title="custom_maze.py"
        # Environment parameters
        env_id: str = "CustomMazeEnv"  # Environment ID
        maze_name: str = "crawford"  # Name of the maze
        state_encoding: str = "binary" # State encoding: binary, onehot, integer
        n: int = 3  # Number of rows in the maze if the maze name is crawford
        P: float = -100 # Value of penalty
        R: float = 100 # Value of reward
        N: float = 0 # Default / neutral reward for all other states
```


