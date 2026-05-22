# Autonomous Over-the-Board Chess Robot
A robot that plays physical chess against a human autonomously
The robot detects board state, computes legal moves using a chess engine, and physically executes moves on a real chessboard.


## Features
- Autonomous over-the-board gameplay
- Physical piece movement
- Legal move generation
- Board state tracking
- Human move detection
- Move validation
- Check/checkmate/stalemate handling
- Integrated chess engine support
- Real-time robot control with Arduino


## Architecture
The system consists of four major components:

1. Board State Detection
2. Chess Engine / Move Generation
3. Path Planning
4. Robotic Motion Control


### Board Detection
The program tracks piece from the starting position and updates board state after each human move is detected using OpenCV characteristic based detection.

### Chess Engine
Legal moves are generated internally and evaluated by the engine to select the strongest move.

### Motion Planning
The robot converts chess coordinates into physical movement commands while avoiding collisions with other pieces.

### Hardware Control
An Arduino controls the robotic arms and executes movement instructions sent from Python.
