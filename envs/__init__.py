from gymnasium.envs.registration import register
import numpy as np
import itertools


try:
    import gym_gridverse
except ImportError:
    print(
        "WARNING: ``gym_gridverse`` is not installed. This means you cannot run an experiment with the gv_*.yaml domains."
    )

try:
    import gym_pomdps
except ImportError:
    print(
        "WARNING: ``gym_pomdps`` is not installed. This means you cannot run an experiment with the HeavenHell or "
        "Hallway domain. "
    )

try:
    import minihack
except ImportError:
    print(
        "WARNING: ``mini_hack`` is not installed. This means you cannot run an experiment with any of the MH- domains."
    )


################
# MEMORY CARDS #
################

register(
    id="Memory-5-v0",
    entry_point="envs.memory_cards:Memory",
    kwargs={"num_pairs": 5},
    max_episode_steps=50,
)


############
# CAR FLAG #
############

register(
    id="DiscreteCarFlag-v0",
    entry_point="envs.car_flag:CarFlag",
    kwargs={"discrete": True},
    max_episode_steps=200,
)


#############
# MINI HACK #
#############

register(
    id="MH-Room-5-v0",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-5x5-v0"},
    max_episode_steps=100,
)

register(
    id="MH-Room-5-v1",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-5x5-v0", "obs_crop": 3},
    max_episode_steps=100,
)

register(
    id="MH-Room-5-v2",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-5x5-v0", "obs_type": "pixel_crop"},
    max_episode_steps=100,
)

register(
    id="MH-DarkRoom-5-v0",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-Dark-5x5-v0"},
    max_episode_steps=100,
)

register(
    id="MH-DarkRoom-5-v1",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-Dark-5x5-v0", "obs_crop": 3},
    max_episode_steps=100,
)

register(
    id="MH-DarkRoom-5-v2",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-Dark-5x5-v0", "obs_type": "pixel_crop"},
    max_episode_steps=100,
)

register(
    id="MH-Room-15-v0",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-15x15-v0"},
    max_episode_steps=300,
)

register(
    id="MH-Room-15-v1",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-15x15-v0", "obs_crop": 3},
    max_episode_steps=300,
)

register(
    id="MH-Room-15-v2",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-15x15-v0", "obs_type": "pixel_crop"},
    max_episode_steps=300,
)

register(
    id="MH-DarkRoom-15-v0",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-Dark-15x15-v0"},
    max_episode_steps=300,
)

register(
    id="MH-DarkRoom-15-v1",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-Dark-15x15-v0", "obs_crop": 3},
    max_episode_steps=300,
)

register(
    id="MH-DarkRoom-15-v2",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-Dark-15x15-v0", "obs_type": "pixel_crop"},
    max_episode_steps=300,
)

register(
    id="MH-Maze-9-v0",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-MazeWalk-9x9-v0"},
    max_episode_steps=180,
)

register(
    id="MH-Maze-9-v1",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-MazeWalk-9x9-v0", "obs_crop": 3},
    max_episode_steps=180,
)

register(
    id="MH-Maze-9-v2",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-MazeWalk-9x9-v0", "obs_type": "pixel_crop"},
    max_episode_steps=180,
)

register(
    id="MH-MazeMap-9-v0",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-MazeWalk-Mapped-9x9-v0"},
    max_episode_steps=180,
)

register(
    id="MH-MazeMap-9-v1",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-MazeWalk-Mapped-9x9-v0", "obs_crop": 3},
    max_episode_steps=180,
)

register(
    id="MH-MazeMap-9-v2",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-MazeWalk-9x9-v0", "obs_type": "pixel_crop"},
    max_episode_steps=180,
)

des_maze_v0 = """
MAZE: "mylevel", ' '
FLAGS:premapped
GEOMETRY:center,center
MAP
||||||||||||
|.|....|.|.|
|...||.|.|.|
||.|||...|.|
|..|...|||.|
||||.|||...|
|..........|
||||||||||||
ENDMAP
STAIR:(10, 1),down
BRANCH: (1,1,1,1),(2,2,2,2)
"""

register(
    id="MH-maze-v1",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": None, "obs_crop": 3, "des_file": des_maze_v0},
    max_episode_steps=180,
)

register(
    id="MH-maze-v2",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": None, "obs_type": "pixel_crop", "des_file": des_maze_v0},
    max_episode_steps=180,
)


############
# MUSHROOM FOREST #
############


def register_mushroom_forest_envs():
    # Original feature weights
    original_weights = [20, 2.0, -1.0]

    # Generate all permutations of the feature weights
    permutations = list(itertools.permutations(original_weights))

    # Register an environment for each permutation
    for i, perm in enumerate(permutations):
        register(
            id=f"MushroomForest-v{i + 1}",
            entry_point="envs.mushroom_forest:Speaker0MushroomForest",
            kwargs={
                "n_cells": 10,
                "max_features": 3,
                "feature_weights": np.array(perm),
                "max_features_per_cell": 1,
                "message_type_probs" : (0, 0.5, 0.5)  # empty,state,reward

            },
            max_episode_steps=200,
        )

    print(f"Registered {len(permutations)} Mushroom Forest environments")
    return permutations


register_mushroom_forest_envs()

