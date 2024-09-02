## Overview
This project is a simplified implementation of the classic Pacman game using Python and Tkinter for the graphical interface. It features a Pacman agent that navigates a grid, avoiding a ghost while collecting dots. The game logic includes a reinforcement learning aspect where the agent can learn optimal moves based on rewards and penalties.

## Features
- **Pacman Movement:** Controlled by an AI agent or manually through predefined rules.
- **Ghost Movement:** The ghost moves randomly with a bias towards Pacman's position.
- **Dot Collection:** Pacman collects dots scattered across the grid to earn points.
- **Win/Loss Conditions:** The game ends when Pacman collects all dots (win) or is caught by the ghost (loss).
- **Graphical Interface:** The game is visualized using a Tkinter-based GUI.

## Classes and Functions

### `State` Class
The `State` class represents the game state, including:
- Pacman's position
- Ghost's position
- Dot locations
- Walls
- Game dynamics such as checking for wins, losses, and valid moves.

Key methods include:
- `__eq__` and `__hash__`: Used for comparing and hashing states.
- `_possible`: Determines possible moves from a given location.
- `_move`: Updates the state based on an action.
- `_get_actions`: Retrieves all possible actions for Pacman in the current state.

### `Pacman` Class
The `Pacman` class defines the actions Pacman can take. It includes:
- `Action` Enum: Defines possible actions (Up, Down, Left, Right).
- `get_actions`: Retrieves valid actions from a given state.

### `GUI` Class
The `GUI` class handles the graphical interface and game loop. It includes:
- `__init__`: Initializes the game window and board.
- `__iterate`: Advances the game state by one step, updating Pacman's position and the ghost's position.
- `__update_board`: Renders the current state of the game on the GUI.
- `__update_score`: Updates the score display.

### `train` Method
The `train` method runs the agent through multiple episodes, allowing it to learn optimal strategies through reinforcement learning.

## Installation
To run the Pacman game, ensure you have Python installed along with the necessary packages:
```bash
pip install tkinter
Running the Game
To start the game, run the following command:

bash
Copy code
python pacman.py
You can control the Pacman agent's behavior by modifying the agents module, which should define the logic for choosing actions and learning from game states.

Customization
Grid Size: You can change the size of the grid by modifying the size parameter in the State class.
Agent Behavior: Customize the AI behavior by altering the get_action and update methods in your agent class.
Dependencies
Python 3.x
Tkinter (for GUI)
License
This project is created for educational purposes as part of the CIS 521 course at the University of Pennsylvania.

Author
Jonathan Kim

