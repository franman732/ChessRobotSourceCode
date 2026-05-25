# Autonomous Over-the-Board Chess Robot
A robot that plays physical chess against a human autonomously by detecting board state, computing legal moves using a chess engine, and physically executing moves on a real chessboard using 3 robotic arms.


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
Legal moves are generated using PyChess and evaluated by the chess engine to select the strongest move.

### Motion Planning
The robot converts chess coordinates into physical movement commands using 2D inverse kinematics and geometric principles while avoiding collisions with other pieces.

### Hardware Control
An NVIDIA Jetson Orin Nano runs the high-level Python control system and sends movement commands to an Arduino Mega, which controls the robot arms. Devices on the Jetson’s local Wi-Fi network can connect via its IP address to remotely access and control the terminal for debugging.
