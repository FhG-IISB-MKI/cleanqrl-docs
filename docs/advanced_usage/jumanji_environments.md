# Jumanji Environments

The Jumanji environments are integrated via the ```create_jumanji_env``` function, which initializes specific environments like TSP, Knapsack, and Maze from the jumanji.environments module. This function takes an ```env_id``` (e.g., "TSP-v1", "Knapsack-v1", "Maze-v0") and a config dictionary to customize parameters such as ```num_cities```, ```num_items```, or maze dimensions (```num_rows```, ```num_cols```). It uses generators (e.g., ```UniformGenerator``` for TSP, ```RandomGeneratorKnapsack``` for Knapsack) to define problem instances. For unsupported ```env_ids```, it falls back to jumanji.make with a warning about potential missing custom wrappers. 


The function applies standard wrappers like ```JumanjiToGymWrapper```, ```FlattenObservation```, and ```RecordEpisodeStatistics``` before adding custom wrappers, ensuring compatibility with Gymnasium’s API. Note that the ```FlattenObservation``` gym wrapper converts the observation from a dictionary to a numpy array.


```py title="jumanji_wrapper.py"
    env = jumanji.wrappers.JumanjiToGymWrapper(env)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if env_id == "TSP-v1":
        env = JumanjiWrapperTSP(env)
    elif env_id == "Knapsack-v1":
        env = JumanjiWrapperKnapsack(env)
    elif env_id == "Maze-v0":
        env = JumanjiWrapperMaze(env, config)
```

# Jumanji Wrappers

Jumanji environments usually have a large state-space. For a QRL agent to act on these environments, we need to somehow condense the state representation into more manageable sizes that can be encoded into quantum cicuits. We do that by creating individual wrappers for each environment. In this repository, there are three custom wrappers ```JumanjiWrapperTSP```, ```JumanjiWrapperKnapsack```, and ```JumanjiWrapperMaze``` for 3 different Jumanji environments. For additional Jumanji environments, you will need to write your own wrapper, which should not be difficult if you follow a similar method to ours.

* ```JumanjiWrapperTSP```: Wraps the TSP environment, overriding ```step``` to track episode counts and compute additional metrics every 100 episodes when truncated. It extracts node coordinates from the state and calculates the optimal tour length using ```tsp_compute_optimal_tour```, which exhaustively tests permutations via ```itertools.permutations```. The ```tsp_compute_tour_length``` helper computes Euclidean distances between nodes, including the return to the start. The wrapper adds ```optimal_tour_length``` and ```approximation_ratio``` to the info dict and is penalizing incomplete tours. Note that the calculation of the ```optimal_tour_length``` is done in brute force, which will be very expensive for large problem sizes.
* ```JumanjiWrapperKnapsack```: Wraps the Knapsack environment, enhancing step to compute optimal values every 100 episodes upon truncation. It uses ```knapsack_optimal_value```, a dynamic programming solution that discretizes weights and values (scaled by precision=1000) to solve the ```0-1 knapsack problem``` efficiently. The wrapper stores the previous state and adds ```optimal_value``` and ```approximation_ratio``` to info, enabling performance evaluation.
* ```JumanjiWrapperMaze```: Wraps the Maze environment, adding a configurable ```reset``` method that supports a constant maze layout via a fixed seed if ```constant_maze``` is ```True``` in the config. The ```step``` method simplifies ```termination``` handling, always returning ```terminate=False``` and clearing info unless truncated. This wrapper is minimal but allows for consistent maze testing.

These wrappers are applied in ```create_jumanji_env``` based on the ```env_id```, building on Jumanji’s base environments and Gymnasium’s utilities to provide a robust RL experimentation framework.