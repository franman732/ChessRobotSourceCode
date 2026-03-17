import chess
import chess.engine
import time

import cv2
from matplotlib import image
import numpy as np
import xarm
import serial
import ast

from datetime import datetime

def print_timestamped(*objects, sep=' ', end='\n', file=None, flush=False):
    # Create timestamp like " 9:25:15 PM" (fixed width)
    timestamp = datetime.now().strftime("%I:%M:%S %p").lstrip("0")
    timestamp = f"{timestamp:>11}"  # force constant width

    # Build message the same way print() would
    message = sep.join(str(obj) for obj in objects)

    output = f"{timestamp} {message}"

    # Print to screen (or provided file)
    print(output, end=end, file=file, flush=flush)

    # Also append to out.txt
    with open("out.txt", "a", encoding="utf-8") as f:
        print(output, end=end, file=f, flush=flush)

def timed_call(fn, *args, name=None):
    start = time.perf_counter()
    result = fn(*args)
    end = time.perf_counter()
    print_timestamped(f"{name or fn.__name__} took {(end - start) * 1000:.2f} ms")
    return result

arduino = serial.Serial(port='COM8', baudrate=9600, timeout=100) # timeout = 1 means 1 second; 1 is right arm, 0 is left arm, 2 is tradeoff arm.

engine = chess.engine.SimpleEngine.popen_uci("C:\\Patricia\\patricia_v3.exe") #  Stockfish\\stockfish-windows-x86-64-avx2"p
board = chess.Board()

BOARD_FLIPPED = True
human = chess.BLACK # This is inverted, I swapped the turns so black goes first, but it is easier just to call the human  black cuz the queen square would need to be flipped somehow
board.turn = human
previous_board = None

width = 750
height = 750
true_corners = np.float32([
    [630, 216], # Top left
    [1250, 229], # Top right
    [1229, 861], # Bottom right
    [622, 849], # Bottom left
])
dst_pts = np.float32([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1],
])

button_clicked = False
not_initialized = True

mean_strength = 1
edges_strength = 1
variance_strength = 1.5
lightness_strength = 1.5
deviation_strength = 1.5
A_strength = 3.25
B_strength = 3.25
threshold = 6
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))

upper_mapx = 14
lower_mapx = 11
upper_mapy = 10
lower_mapy = 8

check_1 = True
previous_positions = None
previous_characteristics = None
initial_change = True # This variable allows the robot to skip the first detection to update its internal values, and then use those values to update baseline values to allow for dynamic changing of initial photos.

cap = None
hist_images = []
previous_hist = None

L1 = 10
L2 = 9.5
L3 = 20.5

servo_baselines_right = [340, 0, 500, 504, 130, 511] # Maybe decrease gripper open value
servo_baselines_left = [340, -50, 500, 509, 134, 500]  #Index 1, servo 2, actually counts as an offset here, not a baseline, but it is easier to just put it here than to add an extra variable for it
default_servo_positions = [400, 495, 0, 274, 850, 511] #This is the default positions the servos reset to each time
tradeoff_arm_default_positions = [0, 493, 862, 150, 140, 491]

Tradeoff_arm_tradeoff_right = [0, 340, 879, 278, 639, 752] # Left and right here are from the perspective of the human
Tradeoff_arm_tradeoff_left = [0, 613, 901, 313, 654, 250] #these are the positions for the tradeoff arm to be in to grab the pieces on the tradeoff platform

Tradeoff_right_arm_positions = [570, 235, 227, 381, 161, 237] # This is the position for the right arm to be in to tradeoff
Tradeoff_left_arm_positions = [0, 700, 225, 444, 121, 765] # This is the position for the left arm to be in to tradeoff

Tradeoff_right_arm_highest_position = [587, 240, 119, 473, 162, 240] # These are the highest positions for the arms to tradeoff so the tradeoff moves vertically and doesnt hit the other arm
tradeoff_left_arm_highest_position = [677, 702, 72, 589, 121, 752] 

tradeoff_waiting_servo_5_position = 350 # Puts shoulder in waiting position to avoid collisions while moving other servos; This is the waiting position for any movements off the board
arduino_delay = 1.1 #This is the delay in milliseconds between sending commands to the Arduino to give it ample time to interpret previous set of commands
        # 1 is right arm, 0 is left arm, 2 is tradeoff arm.
x_offsets = [1.5, .5, -.5, -4.5, -3, 1, .5, 1, 
             -0.5, -1.5, -1.5, -4, -3, -1.5, -1.7, -1.25, 
             -2, -2.5, -3, -4, -3.25, -3, -2.55, -1.9, 
             -2, -3.5, -3.5, -4, -2, -3.15, -2.9, -5, 
             -2.5, -2.9, -3.3, -3.75, -2.4, -3, -3, 0, 
             -2, -2.55, -3, -4, -2, -2.5, -2, -1.8, 
             -.5, -1.7, -1.75, -4, -2, -1, -.5, .25, 
             1.75, 1, -1, -4, -2.5, 1.5, 1.5, 2.5] 

y_offsets = [2.5, 1.25, .5, -1.25, -.5, -3, -1.25, -1.75, 
             0, -1, -1, -2, -.25, -1, .25, .5,
             -1, -1.25, -1.5, -1.5, 0, .5, 1.25, 1.65,
             -1, -1, -.85, -1.25, -1, 0, .25, 1,
             -0.5, -.5, -.75, -.75, 0, 0, 0, 0,
             0, 0, -.5, -.75, 0, 0, 0, -.3,
             -1.25, -1, -1.5, 0, -.5, 0, .5, 1,
             -7, -4.5, -1.75, .25, -.5, 1.75, 2, 4]

target_y_list = [-5.5, -6, -7.5, -7, -6.75, -6.8, -5.3, -4.5,
                -4.5, -5.3, -7, -7, -6.5, -6.5, -4.75, -3.75,
               -4, -5, -6, -7.5, -6.5, -5.25, -4, -3.5,
                -4, -5, -6, -7, -6.5, -5, -4, -2.5,
                -4, -5, -6, -7.5, -6.5, -5, -4, 0,
                -4, -5, -6, -7.5, -6.5, -5.25, -4, -3.4,
                -4.75, -5.5, -6.5, -7.5, -7, -6, -5, -4,
                -5.5, -6.5, -7.25, -8, -7, -6, -5.5, -5.5]

additional_wrist_offsets = [0, 0, +4, +50, +45, 0, 0, 0, 
                            0, 0, 0, +30, +40, 0, 0, 0,
                            0, 0, 0, +20, +12, 0, 0, 0,
                            0, 0, 0, +20, +12, 0, 0, 0,
                            0, 0, 0, +12, 0, 0, 0, 0,
                            0, 0, 0, +15, +30, 0, 0, 0,
                            0, 0, 0, +20, +50, 0, 0, 0,
                            0, 0, +20, +40, +55, +55, 0, 0] # 60

capture_positions_right = [[589, 580, 260, 307, 178, 182], # starting from position number 10, left side starts at 1 and goes to 9, right side starts at 10 and goes to 18
                           [589, 561, 219, 419, 120, 172], #top is closest to board, then going up starting left to right
                           [589, 499, 219, 420, 122, 135],
                           [589, 564, 289, 389, 122, 164],
                           [589, 524, 322, 379, 123, 121]]

capture_positions_left = [[708, 378, 266, 313, 172, 820],
                          [708, 393, 245, 396, 131, 820],
                          [708, 446, 246, 395, 131, 866],
                          [708, 447, 293, 398, 120, 871],
                          [708, 451, 335, 367, 132, 830]]

capture_positions_transfer_right = [[600, 471, 820, 185, 597, 860], #closest
                                   [601, 466, 882, 360, 703, 892],
                                   [599, 474, 890, 498, 779, 890]] # furthest

capture_positions_transfer_left = [[601, 473, 872, 264, 638, 135], #closest
                                    [601, 474, 937, 466, 755, 134],
                                    [600, 474, 898, 478, 769, 133]] #furthest

# Numbers to fix: 61

#special_offsets = {4: -25, 28: -12, 44: -25, 52: -40, 61: -40, 60: -12} # Only for elbow tilts at extremes

default_x_offset = 9
square_size = 3.3

piece_values = { "P": 1, "R": 5, "N": 3, "B": 3, "Q": 9, "K": 9, "p": 1, "r": 5, "n": 3, "b": 3, "q": 9, "k": 9 }

def rc_to_square(row, col):
    if BOARD_FLIPPED:
        # camera top = Black's 8th rank
        return chess.square(7 - col, row)
    else:
        # camera top = White's 8th rank
        return chess.square(col, 7 - row)
    
def square_to_rc(square):
    rank = chess.square_rank(square)
    file = chess.square_file(square)

    if BOARD_FLIPPED:
        return rank, 7 - file
    else:
        return 7 - rank, file

def map_range(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def map_range_clamped(x, in_min, in_max, out_min, out_max):
    x = max(min(x, in_max), in_min)
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def hundredth_map_range(x, in_min, in_max, out_min, out_max):
    return round((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min, 4)

def determine_offset(row, col, square_size):
    const_min = 6
    if col <= 3:
        x_min = map_range(col, 0, 3, 0 + const_min, square_size // 4)
        x_max = map_range(col, 0, 3, square_size // 2, 3 * square_size // 4)
    if col >= 4:
        x_min = map_range(col, 4, 7, square_size // 4, square_size // 2)
        x_max = map_range(col, 4, 7, 3 * square_size // 4, square_size - const_min)
    if row <= 3:
        y_min = map_range(row, 0, 3, 0 + const_min, square_size // 4)
        y_max = map_range(row, 0, 3, square_size // 2, 2 * square_size // 4)
    if row >= 4:
        y_min = map_range(row, 4, 7, square_size // 4, square_size // 2)
        y_max = map_range(row, 4, 7, 3 * square_size // 4, square_size - const_min)
    return x_min, x_max, y_min, y_max

def initialize_camera():
    global cap
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    for _ in range(30):
        cap.read()
    return cap

def take_pictures():
    photos = []
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            photos.append(frame)
        else:
            photos.append(None)
    return photos

    
def warp_board(image):
    global true_corners, dst_pts, width, height
    M = cv2.getPerspectiveTransform(true_corners, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

def infer_chess_board(starting_images):
    inferred_images = []
    global hist_images, clahe
    hist_images = []
    for starting_image in starting_images:
        if starting_image is None:
            print_nathan("Error: No image.")
            return None
        else:
            hist_board = np.zeros((8,8,32), dtype=np.float32)
            inferred_board = np.array([[[0 for _ in range(8)] for _ in range(8)] for _ in range(9)], dtype=np.float32)
            warped_image = warp_board(starting_image)
            square_size = warped_image.shape[0] // 8

            gray_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
            hsv_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
            lab_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2LAB)
            blurred_for_variance = cv2.bilateralFilter(gray_image, d=7, sigmaColor=80, sigmaSpace=50)
            blurred_image = cv2.GaussianBlur(hsv_image, (3, 3), 0)
            blurred_for_lab = cv2.GaussianBlur(lab_image, (3, 3), 0)
            equalized_image = cv2.equalizeHist(gray_image)
            edge_image = equalized_image

            for row in range(8):
                for col in range(8):

                    """mapped_row = map_range(row, 0, 7, lower_mapy, upper_mapy)
                    mapped_col = map_range(col, 0, 7, lower_mapx, upper_mapx)
                    """
                    x_start = col * square_size
                    y_start = row * square_size

                    x_min, x_max, y_min, y_max = determine_offset(row, col, square_size)
                    blurred_HSV_square = blurred_image[y_min + y_start:y_max + y_start, x_min + x_start:x_max + x_start]
                    blurred_square = blurred_for_variance[y_min + y_start:y_max + y_start, x_min + x_start:x_max + x_start]
                    lab_square = blurred_for_lab[y_min + y_start:y_max + y_start, x_min + x_start:x_max + x_start]
                    edge_square = edge_image[y_min + y_start:y_max + y_start, x_min + x_start:x_max + x_start]
                    """test_square = warped_image[y_min + y_start:y_max + y_start, x_min + x_start:x_max + x_start]

                    cv2.imshow("Square 6X7", test_square)
                    cv2.waitKey(0)"""
                    #edges = np.count_nonzero(cv2.Canny(edge_square, 100, 225))

                    gx = cv2.Sobel(edge_square, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(edge_square, cv2.CV_32F, 0, 1, ksize=3)
                    grad_mag = np.sqrt(gx*gx + gy*gy)
                    edges = np.mean(grad_mag)

                    L = lab_square[:,:,0].astype(np.float32)
                    L_norm = (L - np.mean(L)) / (np.std(L) + 1e-6)
                    L_feature = np.mean(np.abs(L_norm))

                    variance = np.var(blurred_square)
                    deviation = np.std(blurred_square)
                    h = blurred_HSV_square[:,:,0]
                    s = blurred_HSV_square[:,:,1]
                    hist_h = cv2.calcHist([h], [0], None, [16], [0, 180])
                    hist_s = cv2.calcHist([s], [0], None, [16], [0, 256])
                    hist = np.concatenate([hist_h, hist_s])
                    hist = cv2.normalize(hist, hist).flatten()
                    #L = np.mean(lab_square[:,:,0])
                    mean_A = np.mean(lab_square[:,:,1])
                    mean_B = np.mean(lab_square[:,:,2])
                    std_a = np.std(lab_square[:,:,1])
                    std_b = np.std(lab_square[:,:,2])
                    inferred_board[0][row][col] = edges
                    inferred_board[1][row][col] = variance
                    inferred_board[2][row][col] = deviation
                    inferred_board[3][row][col] = L_feature
                    inferred_board[4][row][col] = mean_A
                    inferred_board[5][row][col] = mean_B
                    inferred_board[6][row][col] = std_a
                    inferred_board[7][row][col] = std_b
                    inferred_board[8][row][col] = np.mean(clahe.apply(lab_square[:,:,0]))
                    hist_board[row, col, :] = hist
            inferred_images.append(inferred_board)
            hist_images.append(hist_board)
    return inferred_images

def check_button():
    # Placeholder for button checking logic
    return False

def initialize_values():
    while not_initialized == True:
        button_clicked = check_button()
        if button_clicked == True:
            global initial_characteristics
            initial_characteristics = infer_chess_board("C:\\Chess_Images\\image_for_initial_dark.jpg")
            not_initialized = False

def determine_positions():
    global current_characteristics, initial_characteristics
    
    photos = take_pictures()
    current_characteristics = infer_chess_board(photos)

    positions = np.array([[['E' for _ in range(8)] for _ in range(8)] for _ in range(5)])
    diff = np.array([[[0 for _ in range(8)] for _ in range(8)] for _ in range(5)], dtype=np.int64)
    true_positions = np.array([['E' for _ in range(8)] for _ in range(8)])
    for i in range(5):
        for row in range(8):
            for col in range(8):
                edges_dif = np.float64(abs((current_characteristics[i][0][row][col] - initial_characteristics[i][0][row][col]) / (initial_characteristics[i][0][row][col] + 1e-5)))
                variance_dif = np.float64(abs((current_characteristics[i][1][row][col] - initial_characteristics[i][1][row][col]) / (initial_characteristics[i][1][row][col] + 1e-5)))
                deviation_dif = np.float64(abs((current_characteristics[i][2][row][col] - initial_characteristics[i][2][row][col]) / (initial_characteristics[i][2][row][col] + 1e-5)))
                Lightness_dif = np.float64(abs((current_characteristics[i][3][row][col] - initial_characteristics[i][3][row][col]) / (initial_characteristics[i][3][row][col] + 1e-5)))
                A_dif = np.float64(abs((current_characteristics[i][6][row][col] - initial_characteristics[i][6][row][col]) / (initial_characteristics[i][6][row][col] + 1e-5)))
                B_dif = np.float64(abs((current_characteristics[i][7][row][col] - initial_characteristics[i][7][row][col]) / (initial_characteristics[i][7][row][col] + 1e-5)))
                total_dif = np.float64(edges_dif * edges_strength + variance_dif * variance_strength + deviation_dif * deviation_strength + Lightness_dif * lightness_strength + A_dif * A_strength + B_dif * B_strength)
                if total_dif > threshold:
                    positions[i][row][col] = 'X'  
                else: 
                    positions[i][row][col] = 'E'
                diff[i][row][col] = total_dif
    #print_timestamped("Differences:", diff)

    for row in range(8):
        for col in range(8):
            x_count = 0
            for i in range(5):
                if positions[i][row][col] == 'X':
                    x_count += 1
            if x_count >= 3:
                true_positions[row][col] = 'X'
            else:
                true_positions[row][col] = 'E'
    print_timestamped("Current positions:\n", true_positions)
    return true_positions

def starting_values():
    global starting_positions
    starting_positions = np.array([['E' for _ in range(8)] for _ in range(8)])
    for row in range(8):
        for col in range(8):
            if row == 0 or row == 1 or row == 6 or row == 7:
                starting_positions[row][col] = 'X'
            else:
                starting_positions[row][col] = 'E'  
    return starting_positions

def determine_capture(current_characteristics, previous_characteristics, current_positions, previous_positions, past_hist, row, col):
    array_hist_p = np.array(past_hist)
    array_hist_n = np.array(hist_images)
    row_2 = None
    col_2 = None
    to_square = None
    move2 = None
    from_square = None
    if previous_positions[row][col] == 'X' and current_positions[row][col] == 'E':               
        from_square = rc_to_square(row, col)
    if from_square is None:
        return True
    for move in board.legal_moves:
        if move.from_square == from_square:
            row_2, col_2 = square_to_rc(move.to_square)
            if current_positions[row_2][col_2] == 'X':
                prev_hist = np.mean(array_hist_p[:, row_2, col_2, :], axis=0)
                curr_hist = np.mean(array_hist_n[:, row_2, col_2, :], axis=0)
                prev_L = np.mean([previous_characteristics[i][8][row_2][col_2] for i in range(5)])
                new_L = np.mean([current_characteristics[i][8][row_2][col_2] for i in range(5)])
                prev_A = np.mean([previous_characteristics[i][4][row_2][col_2] for i in range(5)])
                new_A = np.mean([current_characteristics[i][4][row_2][col_2] for i in range(5)])
                prev_B = np.mean([previous_characteristics[i][5][row_2][col_2] for i in range(5)])
                new_B = np.mean([current_characteristics[i][5][row_2][col_2] for i in range(5)])
                to_square = rc_to_square(row_2, col_2)
                move2 = chess.Move(from_square, to_square)
                if ((cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA) > .1) and ((abs(prev_L - new_L) > 20) or ((abs(prev_A - new_A) > 2) and (abs(prev_B - new_B) > 2)))):
                    piece = board.piece_at(from_square)
                    if piece and piece.piece_type == chess.PAWN and ((row_2 == 0) or (row_2 == 7)):
                        move2 = chess.Move(from_square, to_square, promotion=chess.QUEEN)
                        print_timestamped("Pawn promotion capture:", move2)
                    if move2 in board.legal_moves:
                        try:
                            if (previous_positions[row][col] == 'X') and (current_positions[row][col] == 'E'):
                                print_timestamped("Detected capture move:", move2)
                                board.push(move2)
                                print_timestamped(board)
                                return False
                            else:           
                                print_timestamped("False capture")
                        except Exception as e:
                            print_timestamped("broken move:", e)
                    else:
                        print_timestamped("not legal: ", move2)
                else:
                    print_timestamped("Mean not strong enough for capture: ", )
                    print_timestamped("Hist: ", cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA))
                    print_timestamped("L: ", abs(prev_L - new_L), "A: ", abs(prev_A - new_A), "B: ", abs(prev_B - new_B))
                    print_timestamped("move: ", move2)
    return True

def detect_castle(current_positions, previous_positions, changed_coordinates):
    king = 0
    rook = 0
    KING = 0
    ROOK = 0
    rook_square = None
    king_square = None
    ROOK_SQUARE = None
    KING_SQUARE = None
    two_change = 0
    move = None
    if board.turn == chess.WHITE:
        turn = "white"
    else:
        turn = "black"
    for position in changed_coordinates:
        row, col = position
        square = rc_to_square(row, col)
        piece = board.piece_at(square)
        if turn == "black":
            if ((previous_positions[position[0]][position[1]] == 'X' and current_positions[position[0]][position[1]] == 'E') and piece is not None and piece.piece_type == chess.KING):
                king = 1
                king_square = square
            if ((previous_positions[position[0]][position[1]] == 'X' and current_positions[position[0]][position[1]] == 'E') and piece is not None and piece.piece_type == chess.ROOK):
                rook = 1
                rook_square = square
            if ((previous_positions[position[0]][position[1]] == 'E' and current_positions[position[0]][position[1]] == 'X') and board.piece_at(square) is None):
                two_change += 1
        if turn == "white":
            if ((previous_positions[position[0]][position[1]] == 'X' and current_positions[position[0]][position[1]] == 'E') and piece is not None and piece.piece_type == chess.KING):
                KING = 1
                KING_SQUARE = square
            if ((previous_positions[position[0]][position[1]] == 'X' and current_positions[position[0]][position[1]] == 'E') and piece is not None and piece.piece_type == chess.ROOK):
                ROOK = 1
                ROOK_SQUARE = square
            if ((previous_positions[position[0]][position[1]] == 'E' and current_positions[position[0]][position[1]] == 'X') and board.piece_at(square) is None):
                two_change += 1
    if king == 1 and rook == 1 and two_change == 2 and (king_square == 60):
        if rook_square == 56:
            move = chess.Move(60,58)
        elif rook_square == 63:
            move = chess.Move(60, 62)
    if KING == 1 and ROOK == 1 and two_change == 2 and (KING_SQUARE == 4):
        if ROOK_SQUARE == 0:
            move = chess.Move(4, 2)
        elif ROOK_SQUARE == 7:
            move = chess.Move(4, 6)
    if move in board.legal_moves:
        board.push(move)
        print_timestamped(board)
        print_timestamped("Detected castle move:", move)
        return False
    else:
        print_timestamped("Not legal castle: ", move, "two_change:", two_change, "king:", king, "rook:", rook, "KING:", KING, "ROOK:", ROOK)
        print_timestamped("rook_square:", rook_square, "king_square:", king_square, "ROOK_SQUARE:", ROOK_SQUARE, "KING_SQUARE:", KING_SQUARE)
        return True

def detect_passant(current_positions, previous_positions, changed_coordinates):
    from_square = to_square = None
    move = None
    move_1 = 0
    pawn_row_1 = pawn_col_1 = pawn_row_2 = pawn_col_2 = end_col = None
    for position in changed_coordinates:
        row, col = position
        square = rc_to_square(row, col)
        piece = board.piece_at(square)
        if ((previous_positions[position[0]][position[1]] == 'X' and current_positions[position[0]][position[1]] == 'E') and piece is not None and piece.piece_type == chess.PAWN):
            if move_1 == 0:
                pawn_row_1 = position[0]
                pawn_col_1 = position[1]
                move_1 += 1
            else:
                pawn_row_2 = position[0]
                pawn_col_2 = position[1]
        if ((previous_positions[position[0]][position[1]] == 'E' and current_positions[position[0]][position[1]] == 'X') and piece is None):
            to_square = square
            end_col = position[1]
    if pawn_row_1 is None or pawn_row_2 is None or pawn_col_1 is None or pawn_col_2 is None or to_square is None:
        print_timestamped("Could not determine passant squares.")
        return True
    if pawn_row_1 == pawn_row_2:
        if pawn_col_1 == end_col:
            from_square = rc_to_square(pawn_row_1, pawn_col_2)
        elif pawn_col_2 == end_col:
            from_square = rc_to_square(pawn_row_2, pawn_col_1)
    if from_square is not None:
        move = chess.Move(from_square, to_square)
    if move in board.legal_moves:
        board.push(move)
        print_timestamped("Detected en passant move:", move)
        print_timestamped(board)
        return False
    print_timestamped("Not legal en passant:", move)
    return True

def check_for_move():
    global check_1, previous_positions, previous_characteristics, hist_images, previous_hist, initial_characteristics, previous_board, human
    skip_move = 0
    changed_coordinates = []
    if check_1 == True:
        previous_positions = starting_values()
        previous_board = board.copy()
        check_1 = False
    current_positions = determine_positions()
    for row in range(8):
        for col in range(8):
            if previous_positions[row][col] != current_positions[row][col]:
                changed_coordinates.append((row, col))
    if len(changed_coordinates) == 2:
        to_square = None
        from_square = None
        for coord in changed_coordinates:
            r, c = coord
            if previous_positions[r][c] == 'E' and current_positions[r][c] == 'X':
                to_square = rc_to_square(r, c)
            elif previous_positions[r][c] == 'X' and current_positions[r][c] == 'E':
                from_square = rc_to_square(r, c)
        if from_square is None or to_square is None:
            print_timestamped("Could not determine from/to squares.")
            skip_move = 1
            return
        else:
            move = chess.Move(from_square, to_square)
        if move in board.legal_moves:
            piece = board.piece_at(from_square)
            print_timestamped("Detected move:", move)
            row_2, col_2 = square_to_rc(to_square)
            if piece and piece.piece_type == chess.PAWN and ((row_2 == 0) or (row_2 == 7)):
                move2 = chess.Move(from_square, to_square, promotion=chess.QUEEN)
                board.push(move2)
                print_timestamped(board)
                print_timestamped("Pawn promotion")
                return
            try:
                board.push(move)
                print_timestamped(board)
            except Exception as e:
                print_timestamped("broken move:", e)
                skip_move = 1
        else:
            print_timestamped("Illegal move detected!", move)
            skip_move = 1
    elif len(changed_coordinates) == 0:
        print_timestamped("no changes")
    elif len(changed_coordinates) == 1:
        print_timestamped("Single change detected")
        if determine_capture(current_characteristics, previous_characteristics, current_positions, previous_positions, previous_hist, changed_coordinates[0][0], changed_coordinates[0][1]):
            print_timestamped("No bueno")
            skip_move = 1
    elif len(changed_coordinates) == 3:
        if detect_passant(current_positions, previous_positions, changed_coordinates):
            print_timestamped("Fake Passant")
            skip_move = 1
    elif len(changed_coordinates) == 4:
        if detect_castle(current_positions, previous_positions, changed_coordinates):
            print_timestamped("Fake Castle")
            skip_move = 1
    else:
        print_timestamped("Multiple changes detected, unable to determine move.")
        skip_move = 1
    if skip_move == 0:
        print_timestamped("We keeping dis one")
        previous_positions = current_positions
        previous_characteristics = current_characteristics
        previous_hist = hist_images # This is just a bunch of reasons the game would end. 
        if (previous_board != board) and (board.is_checkmate() == False) and (board.is_stalemate() == False) and (board.turn != human):
            return True #comment this
        else:
            return False


def determine_move():
    engine.analyse(board, chess.engine.Limit(time=0.1))
    result = engine.play(board, chess.engine.Limit(time=1))
    print_timestamped("Best move:", result.move)
    from_square = chess_square_to_robot_index(result.move.from_square)
    to_square = chess_square_to_robot_index(result.move.to_square)
    return from_square, to_square, result.move

def send_to_arduino(data, label, list = True, not_first = False): # 1 is right arm, 0 is left arm
    resend_attempts = 0
    print_timestamped("label:", label, "Sending to Arduino:", data)
    if list == True:
        arduino.write((",".join(map(str, data)) + "\n").encode())
        message = arduino.readline().decode().strip()
        if message:
            received_list = ast.literal_eval(message)
            print_timestamped("Received:", message)
            #print_timestamped("data:", data[:6])
            if received_list == data[:6]: # 6 to not include arm number at end of data
                print_timestamped("Arduino acknowledged receipt of data.")
            else:
                if resend_attempts < 3:
                    print_timestamped("Arduino response does not match sent data. Resending.")
                    send_to_arduino(data, label)
                else:
                    print_timestamped("Arduino did not acknowledge receipt of data after 3 attempts.")
        else:
            print_timestamped("No message received")
    else:
        arduino.write(bytes(f"{data},", 'utf-8')) # Just in case I want to send something that is not a list
        message = arduino.readline().decode().strip()
        print_timestamped("Received:", message)

def determine_arm_to_use(square): # 1 is right arm, 0 is left arm
    file = square % 8
    if file <= 3:
        return 0 # This is swapped, it was originally 0 1 before I changed the board offsets too
    else: 
        return 1
    
def chess_square_to_robot_index(square):
    file = chess.square_file(square)
    rank = chess.square_rank(square)

    robot_row = rank
    robot_col = 7 - file

    return robot_row * 8 + robot_col

def move_to_exchange_position(arm, type): #Once at default position, moves arms to exchange position for pick or place, and returns to default, type is "pick" or "place"
    print_timestamped("move_to_exchange_position: arm:", arm, "type:", type)
    global default_servo_positions,  Tradeoff_left_arm_positions, Tradeoff_right_arm_positions, Tradeoff_right_arm_highest_position, tradeoff_left_arm_highest_position
    positions = default_servo_positions.copy()
    positions.append(arm)

    if type == "place":
        gripper = 1000
    elif type == "pick":
        gripper = 0

    if arm == 1:
        tradeoff_positions = Tradeoff_right_arm_positions.copy()
        highest_positions = Tradeoff_right_arm_highest_position.copy()
    elif arm == 0:
        tradeoff_positions = Tradeoff_left_arm_positions.copy()
        highest_positions = tradeoff_left_arm_highest_position.copy()

    positions[0] = gripper
    positions[1] = tradeoff_positions[1] # Moves shoulder to tradeoff position
    positions[2] = highest_positions[2] # Moves elbow to tradeoff position
    positions[3] = highest_positions[3] # Moves wrist tilt to tradeoff position
    positions[4] = tradeoff_waiting_servo_5_position # Puts shoulder in waiting position to avoid collisions while moving other servos
    positions[5] = highest_positions[5] # Moves bottom servo to tradeoff position
    send_to_arduino(positions, "exchange_positions")
    time.sleep(1.1)

    positions[4] = highest_positions[4] # Moves shoulder to final position and puts down piece
    send_to_arduino(positions, "exchange_positions")
    time.sleep(1.1)

    positions[1] = tradeoff_positions[1] # Moves shoulder to tradeoff position
    positions[2] = tradeoff_positions[2] # Moves elbow to tradeoff position
    positions[3] = tradeoff_positions[3] # Moves wrist tilt to tradeoff position
    positions[4] = tradeoff_positions[4] # Moves shoulder to tradeoff position
    positions[5] = tradeoff_positions[5] # Moves bottom servo to tradeoff position
    send_to_arduino(positions, "exchange_positions")
    time.sleep(1.1)

    gripper = abs(gripper - 1000) # Opens gripper if placing, closes gripper if picking

    positions[0] = gripper # Opens gripper if placing, closes gripper if picking
    send_to_arduino(positions, "exchange_positions")
    time.sleep(1.1)

    positions[1] = tradeoff_positions[1] # Moves shoulder back to highest position to avoid collisions while moving other servos
    positions[2] = highest_positions[2] # Moves elbow back to highest position to
    positions[3] = highest_positions[3] # Moves wrist tilt back to highest position to avoid collisions while moving other servos
    positions[4] = highest_positions[4] # Moves shoulder back to waiting position to avoid collisions while moving other servos
    positions[5] = highest_positions[5] # Moves bottom servo back to highest position to avoid collisions while moving other servos
    send_to_arduino(positions, "exchange_positions")
    time.sleep(1.1)

    positions[4] = tradeoff_waiting_servo_5_position # Moves shoulder back to waiting position to avoid collisions while moving other servos
    send_to_arduino(positions, "exchange_positions")
    time.sleep(1.1)

    positions[0] = gripper # Resets gripper to default position
    positions[1] = default_servo_positions[1] # Moves wrist rotation to default position
    positions[2] = default_servo_positions[2] # Moves wrist tilt back to default
    positions[3] = default_servo_positions[3] # Moves moves elbow back to default position
    positions[4] = default_servo_positions[4] # Moves shoulder back to default position
    positions[5] = default_servo_positions[5] # Moves bottom servo back to default position
    send_to_arduino(positions, "exchange_positions")
    time.sleep(arduino_delay)

def tradeoff_arm_grab(from_arm, type): # Makes tradeoff arm grab or place piece, starting from default position and ending at default position
    print_timestamped("tradeoff_arm_grab: from_arm:", from_arm, "type:", type)
    global tradeoff_arm_default_positions, Tradeoff_arm_tradeoff_left, Tradeoff_arm_tradeoff_right
    if from_arm == 1:
        tradeoff_positions = Tradeoff_arm_tradeoff_right.copy()
    elif from_arm == 0:
        tradeoff_positions = Tradeoff_arm_tradeoff_left.copy()

    if type == "pickup":
        gripper = 0
    elif type == "place":
        gripper = 1000

    positions = tradeoff_arm_default_positions.copy()
    positions.append(2) # This is just to indicate to the Arduino that this is a tradeoff move and not a normal move

    positions[5] = tradeoff_positions[5] # Moves shoulder to tradeoff position
    positions[3] = tradeoff_positions[3] # Moves wrist tilt to tradeoff position
    positions[2] = tradeoff_positions[2] # Moves elbow to tradeoff position
    positions[1] = tradeoff_positions[1] # Moves shoulder to tradeoff position
    positions[0] = gripper # Sets gripper position based on type

    send_to_arduino(positions, "tradeoff_arm_grab")
    time.sleep(1.1)

    positions[4] = tradeoff_positions[4] # Moves shoulder to tradeoff position
    send_to_arduino(positions, "tradeoff_arm_grab")
    time.sleep(1.5)

    positions[0] = abs(gripper - 1000) # closes gripper to grab piece
    send_to_arduino(positions, "tradeoff_arm_grab")
    time.sleep(1.25)

    positions[4] = tradeoff_arm_default_positions[4] # Moves shoulder to default position to allow for rotating arm
    send_to_arduino(positions, "tradeoff_arm_grab")
    time.sleep(1.1)

    positions[1] = tradeoff_arm_default_positions[1] # Moves shoulder to default position
    positions[2] = tradeoff_arm_default_positions[2] # Moves elbow to default position
    positions[3] = tradeoff_arm_default_positions[3] # Moves wrist tilt to default position
    positions[5] = tradeoff_arm_default_positions[5] # Moves bottom servo to default position
    send_to_arduino(positions, "tradeoff_arm_grab")
    time.sleep(1.1)

def move_tradeoff_arm(from_arm, to_arm): # Moves arm to tradeoff position for pick or place, type is "pick" or "place"
    global tradeoff_arm_default_positions
    tradeoff_arm_grab(from_arm, "pickup")
    tradeoff_arm_grab(to_arm, "place")
    positions = tradeoff_arm_default_positions.copy()
    positions[0] = 1000
    positions.append(2)
    send_to_arduino(positions, "move_tradeoff_arm")
    time.sleep(1.1)


def exchange_pieces(from_arm, to_arm): # Starts with arm having piece already picked upf
    print_timestamped("Exchanging pieces from arm", from_arm, "to arm", to_arm)
    global default_servo_positions
    move_to_exchange_position(from_arm, "place")
    time.sleep(0.5)
    move_tradeoff_arm(from_arm, to_arm)
    time.sleep(0.5)
    move_to_exchange_position(to_arm, "pick") # ends with arm in default position, with exchanged pieces
    time.sleep(0.5)

def determine_x_y(square_number, arm): # 1 is right arm, 0 is left arm
    column = square_number % 8
    row = square_number // 8

    if arm == 1:
        x1 = (7 - column) *square_size + default_x_offset + x_offsets[square_number] #default_x_offset shifts the whole grid cause arm is offset right, x_offsets shifts each column to better fit the square
        y1 = (row - 3) * square_size - square_size / 2 + y_offsets[square_number] #y_offsets shifts each row to better fit the square
    elif arm == 0:
        x1 = column * square_size + default_x_offset + square_size // 2 + x_offsets[square_number] #default_x_offset shifts the whole grid cause arm is offset right, x_offsets shifts each column to better fit the square
        y1 = (4 - row) * square_size - square_size / 2 + y_offsets[square_number] #y_offsets shifts each row to better fit the square

    return x1, y1

def determine_servo_6(square_number, arm, servo_baselines):
    x, y = determine_x_y(square_number, arm)
    angle = np.atan2(y, x)
    #print_timestamped("angle:", angle)
    return (map_range_clamped(angle, -np.pi/2, np.pi/2, -333, 333) + servo_baselines[5])

def determine_if_too_far(target_x, target_y, skip):
    global L1, L2
    if skip == True:
        return False
    distance = np.sqrt(target_x**2 + target_y**2)
    return False #distance > (L1 + L2) or distance < abs(L1 - L2) deal with this later it buggs out on bottom corner squares and returns true and breaks itself

def determine_wrist_tilt(servo_5_val, servo_4_val, servo_baselines):
    global L1, L2, L3
    return max(min(129 - ((servo_5_val - servo_baselines[4]) + (servo_4_val - servo_baselines[3])), 1000), 0)

def determine_angles(target_x, target_y, servo_baselines):
    #print_timestamped("target_x:", target_x, "target_y:", target_y)
    gamma = np.arctan2(target_y, target_x) 
    alpha = np.arccos(np.clip((target_x**2 + target_y**2 + L1**2 - L2**2) / (2 * L1 * np.sqrt(target_x**2 + target_y**2)), -1, 1))
    beta = np.arccos(np.clip((L1**2 + L2**2 - target_x**2 - target_y**2) / (2 * L1 * L2), -1, 1))

    Rtheta1 = gamma - alpha
    Ltheta1 = gamma + alpha
    Rtheta2 = np.pi - beta
    Ltheta2 = beta - np.pi

    #print_timestamped("Rtheta1 (shoulder):", np.degrees(Rtheta1), "Ltheta1 (shoulder):", np.degrees(Ltheta1), "Rtheta2 (elbow):", np.degrees(Rtheta2), "Ltheta2 (elbow):", np.degrees(Ltheta2))

    zero_1 = servo_baselines[4]
    zero_2 = servo_baselines[3]

    zero_1_degrees = map_range_clamped(zero_1, 0, 1000, 0, 180)
    zero_2_degrees = map_range_clamped(zero_2, 0, 1000, 0, 270)

    Rdegree1 = (Rtheta1 * 180) / np.pi
    Ldegree1 = (Ltheta1 * 180) / np.pi
    Rdegree2 = (Rtheta2 * 180) / np.pi
    Ldegree2 = (Ltheta2 * 180) / np.pi

    # Everything 1 is shoulder, everything 2 is elbow
    #print_timestamped("Pre_clamp: Rdegree1 (shoulder):", Rdegree1 + zero_1_degrees, "Ldegree1 (shoulder):", Ldegree1 + zero_1_degrees, "Rdegree2 (elbow):", Rdegree2 + zero_2_degrees, "Ldegree2 (elbow):", Ldegree2 + zero_2_degrees)
    RFinal1 = map_range_clamped(zero_1_degrees + Rdegree1, 0 , 180, 0, 1000)
    RFinal2 = map_range_clamped(zero_2_degrees + Rdegree2, 0, 180, 0, 1000)

    LFinal1 = map_range_clamped(zero_1_degrees + Ldegree1, 0 , 180, 0, 1000)
    LFinal2 = map_range_clamped(zero_2_degrees + Ldegree2, 0, 180, 0, 1000)
    #print_timestamped("RFinal1 (shoulder):", RFinal1, "LFinal1 (shoulder):", LFinal1, "RFinal2 (elbow):", RFinal2, "LFinal2 (elbow):", LFinal2)
    return LFinal1, LFinal2, RFinal1, RFinal2

def score_squares(square):
    global piece_values
    horizontal_score = 0
    vertical_score = 0

    up_square = square - 1 if square % 8 != 0 else None
    down_square = square + 1 if square % 8 != 7 else None
    left_square = square - 8 if square >= 8 else None # Flip these because the robot comes from the side of the board inwards
    right_square = square + 8 if square < 56 else None

    left_piece = board.piece_at(chess_square_to_robot_index(left_square)) if left_square is not None else None
    right_piece = board.piece_at(chess_square_to_robot_index(right_square)) if right_square is not None else None
    up_piece = board.piece_at(chess_square_to_robot_index(up_square)) if up_square is not None else None
    down_piece = board.piece_at(chess_square_to_robot_index(down_square)) if down_square is not None else None
    #print_timestamped("left_piece:", left_piece, "right_piece:", right_piece, "up_piece:", up_piece, "down_piece:", down_piece)
    horizontal_score = (piece_values[left_piece.symbol()] if left_piece is not None else 0) + (piece_values[right_piece.symbol()] if right_piece is not None else 0)
    vertical_score = (piece_values[up_piece.symbol()] if up_piece is not None else 0) + (piece_values[down_piece.symbol()] if down_piece is not None else 0)

    if horizontal_score > vertical_score:
        return "horizontal"
    elif vertical_score > horizontal_score:
        return "vertical"
    elif horizontal_score == vertical_score: # Horizontal and vertical refer to the positions of the piece relative to jaws; horizontal means to left and right, vertical means up and down; also works to think as where gap should face
        return "equal"

def determine_wrist_rotation(servo_6_val, direction, square_number): # Type in this case is "empty" or "piece", and movement is "up" or "down" 
        #print_timestamped("direction:", direction, "servo_6_val:", servo_6_val, "potential 1", servo_6_val - 333, "potential 2", servo_6_val + 333)
        if ((square_number % 8 == 0 or square_number % 8 == 7) and (0 < (square_number // 8) < 7)): # If piece is on far left or far right, rotate wrist to face inwards to prevent gripper from breaking on metal
            return servo_6_val

        if direction == "horizontal":
            potential = servo_6_val - 333 # I think this value is wrong; fix later
            if potential < 0:
                return servo_6_val + 333
            else:
                return potential
        if direction == "vertical" or direction == "equal":
            return servo_6_val
                
def move_to_highest_position(target_x, target_height, servo_6_val, square_num, type, arm, servo_baselines, servo_2_direction, vertical_offset): # this function puts the arm at the highest position to then be moved downwards
    global default_servo_positions
    max_target_height = target_height + vertical_offset
    LFinal1, LFinal2, RFinal1, RFinal2 = determine_angles(target_x, max_target_height, servo_baselines)
    positions = []

    if type == "pickup":
        positions.append(default_servo_positions[0])
    elif type == "dropoff":
        positions.append(1000)

    positions.append(determine_wrist_rotation(servo_6_val, servo_2_direction, square_num) + servo_baselines[1]) # wrist rotation
    positions.append(determine_wrist_tilt(LFinal1, LFinal2, servo_baselines) + additional_wrist_offsets[square_num]) # wrist tilt
    positions.append(LFinal2)   # elbow
    positions.append(LFinal1)   # shoulder
    positions.append(servo_6_val) # base rotation
    positions.append(arm) # arm number
    send_to_arduino(positions, "move_to_highest_position")
    time.sleep(1.5)

def move_downwards(target_x, target_height, servo_6_val, square_num, type, arm, servo_baselines, servo_2_direction): # this function goes from highest position down to lowest position
    global default_servo_positions
    new_height = target_height

    positions = []

    LFinal1, LFinal2, RFinal1, RFinal2 = determine_angles(target_x, new_height, servo_baselines)
    if type == "pickup":
        positions.append(default_servo_positions[0]) # gripper
    elif type == "dropoff":
        positions.append(1000) # gripper

    positions.append(determine_wrist_rotation(servo_6_val, servo_2_direction, square_num) + servo_baselines[1]) # wrist rotation
    positions.append(determine_wrist_tilt(LFinal1, LFinal2, servo_baselines) + additional_wrist_offsets[square_num]) # wrist tilt
    positions.append(LFinal2)   # elbow
    positions.append(LFinal1)   # shoulder
    positions.append(servo_6_val) # base rotation
    positions.append(arm) # arm number
    send_to_arduino(positions, "move_downwards")
    time.sleep(1.2)

    if type == "pickup":
        positions[0] = 1000 # closes gripper once down
        send_to_arduino(positions, "move_downwards")

    elif type == "dropoff":
        positions[0] = default_servo_positions[0] # opens gripper once down
        send_to_arduino(positions, "move_downwards")
        
    time.sleep(1.2)
    #print_timestamped("finished moving downwards, now sleeping for 2 seconds TRANSITION TRANSITION TRANSITION TRANSITION TRANSTITION TRANSITION TRANSITION TRANSITION TRANSITION TRANSITION")


"""    for i in range(5):
        positions = []
        new_height -= 1

        LFinal1, LFinal2, RFinal1, RFinal2 = determine_angles(target_x, new_height, servo_baselines)
        if type == "pickup":
            positions.append(default_servo_positions[0]) # gripper
        elif type == "dropoff":
            positions.append(1000) # gripper

        positions.append(determine_wrist_rotation(servo_6_val, "horizontal") + servo_baselines[1]) # wrist rotation
        positions.append(determine_wrist_tilt(LFinal1, LFinal2, servo_baselines) + additional_wrist_offsets[square_num]) # wrist tilt
        positions.append(LFinal2)   # elbow
        positions.append(LFinal1)   # shoulder
        positions.append(servo_6_val) # base rotation
        positions.append(arm) # arm number
        send_to_arduino(positions, "move_downwards")
        time.sleep(1)
    time.sleep(1)

    if type == "pickup":
        positions[0] = 1000 # closes gripper once down
        send_to_arduino(positions, "move_downwards")
    elif type == "dropoff":
        positions[0] = default_servo_positions[0] # opens gripper once down
        send_to_arduino(positions, "move_downwards")
    time.sleep(1.5)
    print_timestamped("finished moving downwards, now sleeping for 2 seconds TRANSITION TRANSITION TRANSITION TRANSITION TRANSTITION TRANSITION TRANSITION TRANSITION TRANSITION TRANSITION")"""


def move_upwards(target_x, target_height, servo_6_val, square_num, type, arm, servo_baselines, servo_2_direction, vertical_offset): # this function goes from lowest position up to highest position
    global default_servo_positions
    new_height = target_height + vertical_offset

    positions = []

    LFinal1, LFinal2, RFinal1, RFinal2 = determine_angles(target_x, new_height, servo_baselines)

    if type == "pickup":
        positions.append(1000) # gripper
    elif type == "dropoff":
        positions.append(default_servo_positions[0]) # gripper

    positions.append(determine_wrist_rotation(servo_6_val, servo_2_direction, square_num) + servo_baselines[1]) # wrist rotation
    positions.append(determine_wrist_tilt(LFinal1, LFinal2, servo_baselines) + additional_wrist_offsets[square_num]) # wrist tilt
    positions.append(LFinal2)   # elbow
    positions.append(LFinal1)   # shoulder
    positions.append(servo_6_val) # base rotation
    positions.append(arm) # arm number
    send_to_arduino(positions, "move_upwards")
    time.sleep(1.2)

"""
    for i in range(5):
        positions = [] 
        new_height += 1

        LFinal1, LFinal2, RFinal1, RFinal2 = determine_angles(target_x, new_height, servo_baselines)
        print_timestamped("new_target_x:", target_x, "new_height:", new_height, "UPWARDS")

        if type == "pickup":
            positions.append(1000) # gripper
        elif type == "dropoff":
            positions.append(default_servo_positions[0]) # gripper

        positions.append(determine_wrist_rotation(servo_6_val, "horizontal") + servo_baselines[1]) # wrist rotation
        positions.append(determine_wrist_tilt(LFinal1, LFinal2, servo_baselines) + additional_wrist_offsets[square_num]) # wrist tilt
        positions.append(LFinal2)   # elbow
        positions.append(LFinal1)   # shoulder
        positions.append(servo_6_val) # base rotation
        positions.append(arm) # arm number
        send_to_arduino(positions, "move_upwards")
        time.sleep(1) """

def move_to_default(arm, type, setup = False):
    global default_servo_positions, tradeoff_arm_default_positions

    if arm == 0 or arm == 1:
        servo_positions = default_servo_positions.copy()
    elif arm == 2:
        servo_positions = tradeoff_arm_default_positions.copy()

    if type == "pickup":
        gripper_position = 1000
    elif type == "dropoff":
        gripper_position = servo_positions[0]

    positions = [
        gripper_position,
        servo_positions[1],
        servo_positions[2],
        servo_positions[3],
        servo_positions[4],
        servo_positions[5],
        arm
    ]
    send_to_arduino(positions, "move_to_default") # 1 is right arm, 0 is left arm

    if setup == False:
        time.sleep(1.25)
    else:
        return

def move_arm_to_square(square_num, type, arm): # This function starts the entire movement section; starts with arm at default, grabs piece, returns to default
    #print_timestamped("move_arm_to_square: we are moving arm:", arm, "to square:", square_num)
    if arm == 1:
        servo_baselines = servo_baselines_right
    elif arm == 0:
        servo_baselines = servo_baselines_left
    target_height = target_y_list[square_num]
    target_x, target_y = determine_x_y(square_num, arm)
    servo_6_val = determine_servo_6(square_num, arm, servo_baselines)
    servo_2_direction = score_squares(square_num)
    vertical_offset = 3 if (square_num % 8 == 0 or square_num % 8 == 7 or square_num % 8 == 1 or square_num % 8 == 6) else 5 # If piece is on far left or far right, use smaller vertical offset to not tear grippers on metal; 5 is default, 3 is exception

    move_to_highest_position(target_x, target_height, servo_6_val, square_num, type, arm, servo_baselines, servo_2_direction, vertical_offset)
    move_downwards(target_x, target_height, servo_6_val, square_num, type, arm, servo_baselines, servo_2_direction)
    #print_timestamped("finished moving downwards, now moving upwards TRANSITION TRANSITION TRANSITION TRANSITION TRANSTITION TRANSITION TRANSITION TRANSITION TRANSITION TRANSITION")
    time.sleep(.5)
    move_upwards(target_x, target_height, servo_6_val, square_num, type, arm, servo_baselines, servo_2_direction, vertical_offset)
    #print_timestamped("finished moving upwards, now moving to default TRANSITION TRANSITION TRANSITION TRANSITION TRANSTITION TRANSITION TRANSITION TRANSITION TRANSITION TRANSITION")
    time.sleep(.5)
    move_to_default(arm, type)

def move_arm_to_remove_piece(arm, side = None): # This function starts from default position with piece, moves to capture position and drops off piece, and then moves back to default position; based off arm passed, it determines capture positions but does not do transfer
    global capture_positions_right, capture_positions_left, default_servo_positions, Tradeoff_arm_tradeoff_left, Tradeoff_arm_tradeoff_right, tradeoff_arm_default_positions
    positions = []
    print_timestamped("move_arm_to_remove_piece: we are removing piece with arm:", arm, "and capture positions are:", capture_positions_right if arm == 1 else capture_positions_left)
    if (arm == 0 or arm == 1):
        default_positions = default_servo_positions.copy()
        positions = default_servo_positions.copy()
        if arm == 1:
            capture_positions = capture_positions_right.pop(0) # Gets next available capture position and removes it from the list
        elif arm == 0:
            capture_positions = capture_positions_left.pop(0)
    elif arm == 2:
        default_positions = tradeoff_arm_default_positions.copy()
        positions = tradeoff_arm_default_positions.copy()
        if side == "right":
            capture_positions = capture_positions_transfer_right.pop(0)
        elif side == "left":
            capture_positions = capture_positions_transfer_left.pop(0)
    print_timestamped("move_arm_to_remove_piece: capture positions are:", capture_positions)
    positions.append(arm)
    positions[0] = 1000 # closes gripper to keep piece grabbed
    positions[1] = capture_positions[1] # Rotates wrist to capture position
    positions[2] = capture_positions[2] # Moves wrist tilt to capture position
    positions[3] = capture_positions[3] # Moves elbow to capture position
    positions[4] = tradeoff_waiting_servo_5_position # Moves shoulder to waiting position to avoid collisions while moving other servos
    positions[5] = capture_positions[5] # Moves bottom servo to capture position

    send_to_arduino(positions, "move_arm_to_remove_piece")
    time.sleep(1.1)

    positions[4] = capture_positions[4] # Moves shoulder to capture position to drop piece off
    send_to_arduino(positions, "move_arm_to_remove_piece")
    time.sleep(1.1)

    positions[0] = default_positions[0] # Opens gripper to drop piece off
    send_to_arduino(positions, "move_arm_to_remove_piece")
    time.sleep(1.1)

    positions[4] = tradeoff_waiting_servo_5_position # Moves shoulder back to default position to avoid collisions while moving other servos
    send_to_arduino(positions, "move_arm_to_remove_piece")
    time.sleep(1.1)

    positions[1] = default_positions[1] # Moves wrist rotation back to default positionq
    positions[2] = default_positions[2] # Moves wrist tilt back to default position
    positions[3] = default_positions[3] # Moves elbow back to default position
    positions[4] = default_positions[4] # Moves shoulder back to default position
    positions[5] = default_positions[5] # Moves bottom servo back to default position
    send_to_arduino(positions, "move_arm_to_remove_piece")
    time.sleep(1.1)

def capture(from_square, from_arm): # Starts from defualt, picks up piece, moves it off board, and then ends at default position
    global capture_positions_right, capture_positions_left, tradeoff_arm_default_positions, default_servo_positions
    move_arm_to_square(from_square, "pickup", from_arm)

    print_timestamped("we are in capture and from_arm is:", from_arm, "and capture positions are:", capture_positions_right if from_arm == 1 else capture_positions_left)
    if from_arm == 1:
        capture_positions = capture_positions_right
    elif from_arm == 0:
        capture_positions = capture_positions_left
    print_timestamped("capture positions are:", capture_positions)
    if len(capture_positions) != 0:
        move_arm_to_remove_piece(from_arm)
        return
    elif len(capture_positions) == 0:
        if from_arm == 1: 
            move_to_exchange_position(1, "place")
            time.sleep(0.5)
            tradeoff_arm_grab(from_arm, "pickup")
            time.sleep(0.5)
            move_arm_to_remove_piece(2, "left")
            time.sleep(0.5)
        elif from_arm == 0:
            move_to_exchange_position(0, "place") # Actual arms put piece in exchange position
            time.sleep(0.5)
            tradeoff_arm_grab(from_arm, "pickup") # Tradeoff Arm picks up piece
            time.sleep(0.5)
            move_arm_to_remove_piece(2, "right") # Tradeoff arm places piece down in capture position
            time.sleep(0.5)
        positions = tradeoff_arm_default_positions.copy()
        positions[0] = 1000
        positions.append(2)
        send_to_arduino(positions, "move_tradeoff_arm")
        time.sleep(1)
    

def determine_type_of_robot_move(move):
    print_timestamped(move)
    if move.promotion is True:
        print_timestamped("Move is promotion:", move, "promotion thingi:", move.promotion)
        return "promotion"
    elif board.is_en_passant(move) is True:
        print_timestamped("Move is en passant:", move, "en passant thingi:", board.is_en_passant(move))
        return "en passant"
    elif board.is_capture(move) is True:
        print_timestamped("Move is capture:", move, "capture thingi:", board.is_capture(move))
        return "capture"
    elif board.is_castling(move) is True:
        print_timestamped("Move is castling:", move, "castling thingi:", board.is_castling(move))
        return "castling"
    else:
        print_timestamped("move is normal:", move)
        return "normal"
    
def determine_if_swap_needed(from_arm, to_arm):
    if from_arm != to_arm:
        return True
    else:
        return False
    
def normal_move(from_square, to_square, from_arm, to_arm, swap): # Starts at default position, moves piece from from_square to to_square, and then ends at default position
    if swap == True:
        move_arm_to_square(from_square, "pickup", from_arm)
        exchange_pieces(from_arm, to_arm)
        move_arm_to_square(to_square, "dropoff", to_arm)
    else:
        move_arm_to_square(from_square, "pickup", from_arm)
        time.sleep(0.5)
        move_arm_to_square(to_square, "dropoff", from_arm)

def en_passant_move(from_square, to_square, from_arm, to_arm, swap): # Starts at default position, removes piece on to_square, moves piece from from_square to to_square, and then ends at default position
    captured_piece_square = None
    captured_piece_square = to_square - 8
    capture(captured_piece_square, to_arm) # If from and to arm are different, its always to arm that captures, but if they are the same it doesnt matter
    if swap == True:
        move_arm_to_square(from_square, "pickup", from_arm)
        exchange_pieces(from_arm, to_arm)
        move_arm_to_square(to_square, "dropoff", to_arm)
    else:
        move_arm_to_square(from_square, "pickup", from_arm)
        time.sleep(0.5)
        move_arm_to_square(to_square, "dropoff", from_arm)

def capture_move(from_square, to_square, from_arm, to_arm, swap): # starts at default position, removes piece on to_square, moves piece from from_square to to_square, and then ends at default position
    capture(to_square, to_arm) # If from and to arm are different, its always to arm that captures, but if they are the same it doesnt matter
    if swap == True:
        move_arm_to_square(from_square, "pickup", from_arm)
        exchange_pieces(from_arm, to_arm)
        move_arm_to_square(to_square, "dropoff", to_arm)
    else:
        move_arm_to_square(from_square, "pickup", from_arm)
        move_arm_to_square(to_square, "dropoff", from_arm)

def castle_move(from_square, to_square, from_arm, to_arm, swap): # starts at default position, moves piece from from_square to to_square, moves rook, and then ends at default position
    if swap == True:
        move_arm_to_square(from_square, "pickup", from_arm)
        exchange_pieces(from_arm, to_arm)
        move_arm_to_square(to_square, "dropoff", to_arm)
    else:
        move_arm_to_square(from_square, "pickup", from_arm)
        move_arm_to_square(to_square, "dropoff", from_arm)

    if to_square > from_square: # queenside castle
        rook_from = to_square + 2
        rook_to = to_square - 1
    else: # kingside castle
        rook_from = to_square - 1
        rook_to = to_square + 1

    rook_from_arm = determine_arm_to_use(rook_from)
    rook_to_arm = determine_arm_to_use(rook_to)
    if swap == True:
        move_arm_to_square(rook_from, "pickup", rook_from_arm)
        exchange_pieces(rook_from_arm, rook_to_arm)
        move_arm_to_square(rook_to, "dropoff", rook_to_arm)
    else:
        move_arm_to_square(rook_from, "pickup", rook_from_arm)
        move_arm_to_square(rook_to, "dropoff", rook_from_arm)
    

def promotion_move(from_square, to_square, from_arm, to_arm, swap, move): # starts at default position, removes piece on to_square if there is one, moves piece from from_square to to_square, promotes piece, and then ends at default position
    if board.is_capture(move):
        capture(to_square, to_arm) # If from and to arm are different, its always to arm that captures, but if they are the same it doesnt matter
    if swap == True:
        move_arm_to_square(from_square, "pickup", from_arm)
        exchange_pieces(from_arm, to_arm)
        move_arm_to_square(to_square, "dropoff", to_arm)
    else:
        move_arm_to_square(from_square, "pickup", from_arm)
        move_arm_to_square(to_square, "dropoff", from_arm)

def initialize_game():
    global initial_characteristics
    initialize_camera()
    initial_characteristics = infer_chess_board([cv2.imread("C:\\Chess_Images\\WIN_20260310_19_48_47_Pro.jpg"),
                                                cv2.imread("C:\\Chess_Images\\WIN_20260310_19_48_48_Pro.jpg"),
                                                cv2.imread("C:\\Chess_Images\\WIN_20260310_19_48_49_Pro.jpg"),
                                                cv2.imread("C:\\Chess_Images\\WIN_20260310_19_48_50_Pro.jpg"),
                                                cv2.imread("C:\\Chess_Images\\WIN_20260310_19_48_51_Pro.jpg")
                                                ])
    starting_values()
    send_to_arduino(10, "initialize game", False, True) # Signal to arduino to move arms to default

initialize_game()

while board.is_checkmate() == False and board.is_stalemate() == False: 
    if check_for_move():
        f, t, m = determine_move() # m is move, f is from square, t is to square
        from_arm = determine_arm_to_use(f)
        to_arm = determine_arm_to_use(t)
        move_type = determine_type_of_robot_move(m)
        swap = determine_if_swap_needed(from_arm, to_arm)

        print_timestamped("Move type:", move_type, "From square:", f, "To square:", t, "From arm:", from_arm, "To arm:", to_arm, "Swap needed:", swap)

        if move_type == "normal":
            print_timestamped("normal move")
            normal_move(f, t, from_arm, to_arm, swap)
        elif move_type == "capture":
            print_timestamped("capture move")
            capture_move(f, t, from_arm, to_arm, swap)
        elif move_type == "en passant":
            print_timestamped("en passant move")
            en_passant_move(f, t, from_arm, to_arm, swap)
        elif move_type == "castling":
            print_timestamped("castle move")
            castle_move(f, t, from_arm, to_arm, swap)
        elif move_type == "promotion":
            print_timestamped("promotion move")
            promotion_move(f, t, from_arm, to_arm, swap, m)
else:
    if board.is_checkmate():
        print_timestamped("Checkmate! Winner:", "Black" if board.turn == "Black" else "White")
    elif board.is_stalemate():
        print_timestamped("Stalemate! It's a draw.")




        """move_arm_to_square(f, "pickup", from_arm)
        if from_arm != to_arm:
            move_to_default(from_arm, "pick")
            exchange_pieces(from_arm, to_arm)
        else: # if the move is within the same arm, arm passed as parameter does not matter; can be to_arm or from_arm
            time.sleep(1.5)
            move_arm_to_square(t, "dropoff", from_arm)
        move_to_default(from_arm, "pickup")"""


"""time.sleep(2) #This is just for testing cuz arduino needs time to link w computer
move_arm_to_square(37, "pickup", 1)


while True:
    check_for_move()"""
"""



original_image = take_pictures()[0]
warped_image = warp_board(original_image)
if original_image is None:
    print_timestamped("Error: Could not capture image.")
cv2.imshow("Original", original_image)
cv2.imshow("Warped", warped_image)
cv2.waitKey(0)
print_timestamped("dimensions:", original_image.shape)
print_timestamped("Intensity: ")
image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
hsv_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
square_size = image.shape[0] // 8


x_start = 1 * square_size
y_start = 5 * square_size
mapped_row = map_range(5, 0, 7, lower_mapy, upper_mapy)
mapped_col = map_range(1, 0, 7, lower_mapx, upper_mapx)
square = warped_image[y_start + mapped_row + square_size // 5:(y_start + square_size) - square_size//5 + mapped_row, x_start + mapped_col + square_size//5:(x_start + square_size) - square_size//5 + mapped_col]
cv2.imshow("Square 1X5", square)


x_start = 7 * square_size
y_start = 7 * square_size
mapped_row = map_range(7, 0, 7, lower_mapy, upper_mapy)
mapped_col = map_range(7, 0, 7, lower_mapx, upper_mapx)
square = warped_image[y_start + mapped_row + square_size // 5:(y_start + square_size) - square_size//5 + mapped_row, x_start + mapped_col + square_size//5:(x_start + square_size) - square_size//5 + mapped_col]
cv2.imshow("Square 7x7", square)

x_start = 7 * square_size
y_start = 0 * square_size
mapped_row = map_range(7, 0, 7, lower_mapy, upper_mapy)
mapped_col = map_range(0, 0, 7, lower_mapx, upper_mapx)
square = warped_image[y_start + mapped_row + square_size // 5:(y_start + square_size) - square_size//5 + mapped_row, x_start + mapped_col + square_size//5:(x_start + square_size) - square_size//5 + mapped_col]
cv2.imshow("Square 7X0", square)

x_start = 0 * square_size
y_start = 7 * square_size
mapped_row = map_range(0, 0, 7, lower_mapy, upper_mapy)
mapped_col = map_range(7, 0, 7, lower_mapx, upper_mapx)
square = warped_image[y_start + mapped_row + square_size // 5:(y_start + square_size) - square_size//5 + mapped_row, x_start + mapped_col + square_size//5:(x_start + square_size) - square_size//5 + mapped_col]
cv2.imshow("Square 0X7", square)

x_start = 3 * square_size
y_start = 0 * square_size
mapped_row = map_range(3, 0, 7, lower_mapy, upper_mapy)
mapped_col = map_range(0, 0, 7, lower_mapx, upper_mapx)
square = warped_image[y_start + mapped_row + square_size // 5:(y_start + square_size) - square_size//5 + mapped_row, x_start + mapped_col + square_size//5:(x_start + square_size) - square_size//5 + mapped_col]
cv2.imshow("Square 3X0", square)

x_start = 7 * square_size
y_start = 1 * square_size
mapped_row = map_range(7, 0, 7, lower_mapy, upper_mapy)
mapped_col = map_range(1, 0, 7, lower_mapx, upper_mapx)
square = warped_image[y_start + mapped_row + square_size // 5:(y_start + square_size) - square_size//5 + mapped_row, x_start + mapped_col + square_size//5:(x_start + square_size) - square_size//5 + mapped_col]
cv2.imshow("Square 7X1", square)

x_start = 6 * square_size
y_start = 6 * square_size
mapped_row = map_range(6, 0, 7, lower_mapy, upper_mapy)
mapped_col = map_range(6, 0, 7, lower_mapx, upper_mapx)
square = warped_image[y_start + mapped_row + square_size // 5:(y_start + square_size) - square_size//5 + mapped_row, x_start + mapped_col + square_size//5:(x_start + square_size) - square_size//5 + mapped_col]
cv2.imshow("Square 6X6", square)

x_start = 7 * square_size
y_start = 6 * square_size
mapped_row = map_range(6, 0, 7, lower_mapy, upper_mapy)
mapped_col = map_range(7, 0, 7, lower_mapx, upper_mapx)
square = warped_image[y_start + mapped_row + square_size // 5:(y_start + square_size) - square_size//5 + mapped_row, x_start + mapped_col + square_size//5:(x_start + square_size) - square_size//5 + mapped_col]
cv2.imshow("Square 7X6", square)

x_start = 6 * square_size
y_start = 7 * square_size
mapped_row = map_range(7, 0, 7, lower_mapy, upper_mapy)
mapped_col = map_range(6, 0, 7, lower_mapx, upper_mapx)
square = warped_image[y_start + mapped_row + square_size // 5:(y_start + square_size) - square_size//5 + mapped_row, x_start + mapped_col + square_size//5:(x_start + square_size) - square_size//5 + mapped_col]
cv2.imshow("Square 6X7", square)
cv2.waitKey(0)







x_start = 4 * square_size
y_start = 4 * square_size
square = image[y_start + map_range(5, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(5, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
blurred_image = cv2.GaussianBlur(square, (5, 5), 0)
hsv = hsv_image[y_start + map_range(5, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(5, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
h, w, _ = hsv.shape
center = cv2.GaussianBlur(hsv[h//4:3*h//4, w//4:3*w//4], (3, 3), 0)
print_timestamped("4, 4: mean intensity:", cv2.mean(center[:, :, 2])[0], ", edges:", np.count_nonzero(cv2.Canny(blurred_image, 75, 225)) / blurred_image.size, ", variance:", np.var(blurred_image), ", Deviation:", np.std(blurred_image))

x_start = 5 * square_size
y_start = 6 * square_size
square = image[y_start + map_range(7, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(6, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
blurred_image = cv2.GaussianBlur(square, (5, 5), 0)
hsv = hsv_image[y_start + map_range(7, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(6, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
h, w, _ = hsv.shape
center = cv2.GaussianBlur(hsv[h//4:3*h//4, w//4:3*w//4], (3, 3), 0)
print_timestamped("5, 6: mean intensity:", cv2.mean(center[:, :, 2])[0], ", edges:", np.count_nonzero(cv2.Canny(blurred_image, 75, 225)) / blurred_image.size, ", variance:", np.var(blurred_image), ", Deviation:", np.std(blurred_image))

x_start = 7 * square_size
y_start = 7 * square_size
square = image[y_start + map_range(8, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(8, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
blurred_image = cv27.GaussianBlur(square, (5, 5), 0)
hsv = hsv_image[y_start + map_range(8, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(8, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
h, w, _ = hsv.shape
center = cv2.GaussianBlur(hsv[h//4:3*h//4, w//4:3*w//4], (3, 3), 0)
print_timestamped("7, 7: mean intensity:", cv2.mean(center[:, :, 2])[0], ", edges:", np.count_nonzero(cv2.Canny(blurred_image, 75, 225)) / blurred_image.size, ", variance:", np.var(blurred_image), ", Deviation:", np.std(blurred_image))

x_start = 0 * square_size
y_start = 0 * square_size
square = image[y_start + map_range(0, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(0, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
blurred_image = cv2.GaussianBlur(square, (5, 5), 0)
hsv = hsv_image[y_start + map_range(0, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(0, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
h, w, _ = hsv.shape
center = cv2.GaussianBlur(hsv[h//4:3*h//4, w//4:3*w//4], (3, 3), 0)
print_timestamped("0, 0: mean intensity:", cv2.mean(center[:, :, 2])[0], ", edges:", np.count_nonzero(cv2.Canny(blurred_image, 75, 225)) / blurred_image.size, ", variance:", np.var(blurred_image), ", Deviation:", np.std(blurred_image))

x_start = 7 * square_size
y_start = 0 * square_size
square = image[y_start + map_range(0, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(8, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
blurred_image = cv2.GaussianBlur(square, (5, 5), 0)
hsv = hsv_image[y_start + map_range(0, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(8, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
h, w, _ = hsv.shape
center = cv2.GaussianBlur(hsv[h//4:3*h//4, w//4:3*w//4], (3, 3), 0)
print_timestamped("7, 0: mean intensity:", cv2.mean(center[:, :, 2])[0], ", edges:", np.count_nonzero(cv2.Canny(blurred_image, 75, 225)) / blurred_image.size, ", variance:", np.var(blurred_image), ", Deviation:", np.std(blurred_image))

x_start = 0 * square_size
y_start = 7 * square_size
square = image[y_start + map_range(8, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(0, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
blurred_image = cv2.GaussianBlur(square, (5, 5), 0)
hsv = hsv_image[y_start + map_range(8, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(0, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
h, w, _ = hsv.shape
center = cv2.GaussianBlur(hsv[h//4:3*h//4, w//4:3*w//4], (3, 3), 0)
print_timestamped("0, 7: mean intensity:", cv2.mean(center[:, :, 2])[0], ", edges:", np.count_nonzero(cv2.Canny(blurred_image, 75, 225)) / blurred_image.size, ", variance:", np.var(blurred_image), ", Deviation:", np.std(blurred_image))



hsv = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
square = hsv[4 * square_size:4 * square_size + square_size, 4 * square_size:4 * square_size + square_size]
cv2.imshow("Square 4x4", square)
avg_v = cv2.mean(hsv[:, :, 2])[0]
print_timestamped("4x4", avg_v)

square = hsv[5 * square_size:4 * square_size + square_size, 6 * square_size:4 * square_size + square_size]
cv2.imshow("Square 5x6", square)
cv2.waitKey(0)
avg_v = cv2.mean(hsv[:, :, 2])[0]
print_timestamped("5x6", avg_v)


x_start = 4 * square_size
y_start = 4 * square_size
square = image[y_start:y_start + square_size, x_start:x_start + square_size]
mean_intensity = cv2.mean(square)[0]
print_timestamped(mean_intensity)
print_timestamped(", ")

x_start = 5 * square_size
y_start = 6 * square_size
square = image[y_start:y_start + square_size, x_start:x_start + square_size]
mean_intensity = cv2.mean(square)[0]
print_timestamped(mean_intensity)

x_start = 4 * square_size
y_start = 4 * square_size
square = image[y_start:y_start + square_size - 2, x_start:x_start + square_size]
cv2.imshow("Square 4,4", square,)
cv2.waitKey(0)

edges = cv2.Canny(blurred_image, 75, 225)
print_timestamped("4, 4: edges:", np.count_nonzero(edges) / edges.size) #, ", variance:", np.var(blurred_image), ", Deviation:", np.std(blurred_image))



x_start = 5 * square_size
y_start = 6 * square_size
square = image[y_start:y_start + square_size + 2, x_start + 1:x_start + square_size]
cv2.imshow("Square 5, 6", square)

blurred_image = cv2.GaussianBlur(square, (5, 5), 0)
edges = cv2.Canny(blurred_image, 75, 225)
cv2.imshow("Edges 5,6", edges)
cv2.waitKey(0)
print_timestamped("6, 7: edges:", np.count_nonzero(edges) / edges.size) #, ", variance:", np.var(blurred_image), ", Deviation:", np.std(blurred_image))
"""
