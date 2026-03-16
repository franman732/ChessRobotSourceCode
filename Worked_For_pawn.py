import chess
import chess.engine
import time

import cv2
from matplotlib import image
import numpy as np
import xarm
import serial

def timed_call(fn, *args, name=None):
    start = time.perf_counter()
    result = fn(*args)
    end = time.perf_counter()
    print(f"{name or fn.__name__} took {(end - start) * 1000:.2f} ms")
    return result

arduino = serial.Serial(port='COM8', baudrate=9600, timeout=1)

engine = chess.engine.SimpleEngine.popen_uci("C:\\Stockfish\\stockfish-windows-x86-64-avx2")
board = chess.Board()

BOARD_FLIPPED = True
human = chess.BLACK
board.turn = human
previous_board = None

width = 750
height = 750
true_corners = np.float32([
    [633, 240], # Top left
    [1252, 245], # Top right
    [1241, 875], # Bottom right
    [631, 871], # Bottom left
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

cap = None
hist_images = []
previous_hist = None

L1 = 10
L2 = 9.5
L3 = 20.5

servo_baselines = [365, 500, 500, 504, 130, 511]
default_servo_positions = [400, 495, 0, 274, 879, 511]

Tradeoff_arm_tradeoff_right = [0, 425, 893, 291, 654, 757] # Left and right here are from the perspective of the human
Tradeoff_arm_tradeoff_left = [0, 425, 893, 297, 660, 248]
Tradeoff_arm_start_position = [621, 490, 225, 444, 121, 750]
tradeoff_arm_tradeoff_default = [621, 490, 225, 444, 121, 750]

Tradeoff_right_arm = []
Tradeoff_left_arm = [0, 490, 225, 444, 121, 750]

x_offsets = [0, 0, 0, 0, -2, 1, .5, 1, 
             0, 0, 0, 0, -4.5, -1.5, -1.7, -1.25, 
             0, 0, 0, 0, -3.25, -3, -2.55, -1.9, 
             0, 0, 0, 0, -2, -3.15, -2.9, -5, 
             0, 0, 0, 0, -2.4, -3, -3, 0, 
             0, 0, 0, 0, -2, -2.5, -2, -1.8, 
             0, 0, 0, 0, -2, -1, -.5, .25, 
             0, 0, 0, 0, 0, 1.5, 1.5, 2.5] 

y_offsets = [0, 0, 0, 0, 0, -3, -1.25, -1.75, 
             0, 0, 0, 0, 0, -1, .25, 1.25,
             0, 0, 0, 0, 0, .5, 1.25, 1.65,
             0, 0, 0, 0, -1, 0, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, -.3,
             0, 0, 0, 0, -.5, 0, .5, 1,
             0, 0, 0, 0, 0, 1.75, 2, 5]

target_y_list = [0, 0, 0, 0, -7, -6.8, -5.3, -4.5,
                0, 0, 0, 0, -6.5, -6.5, -4.75, -3.75,
                0, 0, 0, 0, -6.5, -5.25, -4, -3.5,
                0, 0, 0, 0, -6.5, -5, -4, -2.5,
                0, 0, 0, 0, -6.5, -5, -4, 0,
                0, 0, 0, 0, -6.5, -5.25, -4, -3.4,
                0, 0, 0, 0, -7, -6, -5, -4,
                0, 0, 0, 0, 0, -6, -5.5, -4.5]

additional_wrist_offsets = [0, 0, 0, 0, +40, 0, 0, 0, #4
                            0, 0, 0, 0, +40, 0, 0, 0,
                            0, 0, 0, 0, +12, 0, 0, 0,
                            0, 0, 0, 0, +12, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, +30, 0, 0, 0,
                            0, 0, 0, 0, +50, 0, 0, 0,
                            0, 0, 0, 0, 0, +55, 0, 0] # 60

# Numbers to fix: 61

special_offsets = {4: -25, 28: -12, 44: -25, 52: -40, 61: -40, 60: -12} # Only for elbow tilts at extremes

default_x_offset = 9
square_size = 3.3

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
            print("Error: No image.")
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
    print("Differences:", diff)

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
    print("Current positions:\n", true_positions)
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
                        print("Pawn promotion capture:", move2)
                    if move2 in board.legal_moves:
                        try:
                            if (previous_positions[row][col] == 'X') and (current_positions[row][col] == 'E'):
                                print("Detected capture move:", move2)
                                board.push(move2)
                                print(board)
                                return False
                            else:           
                                print("False capture")
                        except Exception as e:
                            print("broken move:", e)
                    else:
                        print("not legal: ", move2)
                else:
                    print("Mean not strong enough for capture: ", )
                    print("Hist: ", cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA))
                    print("L: ", abs(prev_L - new_L), "A: ", abs(prev_A - new_A), "B: ", abs(prev_B - new_B))
                    print("move: ", move2)
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
    if KING == 1 and ROOK == 1 and two_change == 2 and (KING_SQUARE == 3):
        if ROOK_SQUARE == 0:
            move = chess.Move(3, 1)
        elif ROOK_SQUARE == 7:
            move = chess.Move(3, 5)
    if move in board.legal_moves:
        board.push(move)
        print(board)
        print("Detected castle move:", move)
        return False
    else:
        print("Not legal castle: ", move, "two_change:", two_change, "king:", king, "rook:", rook, "KING:", KING, "ROOK:", ROOK)
        print("rook_square:", rook_square, "king_square:", king_square, "ROOK_SQUARE:", ROOK_SQUARE, "KING_SQUARE:", KING_SQUARE)
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
        print("Could not determine passant squares.")
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
        print("Detected en passant move:", move)
        print(board)
        return False
    print("Not legal en passant:", move)
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
            print("Could not determine from/to squares.")
            skip_move = 1
            return
        else:
            move = chess.Move(from_square, to_square)
        if move in board.legal_moves:
            piece = board.piece_at(from_square)
            print("Detected move:", move)
            row_2, col_2 = square_to_rc(to_square)
            if piece and piece.piece_type == chess.PAWN and ((row_2 == 0) or (row_2 == 7)):
                move2 = chess.Move(from_square, to_square, promotion=chess.QUEEN)
                board.push(move2)
                print(board)
                print("Pawn promotion")
                return
            try:
                board.push(move)
                print(board)
            except Exception as e:
                print("broken move:", e)
                skip_move = 1
        else:
            print("Illegal move detected!", move)
            skip_move = 1
    elif len(changed_coordinates) == 0:
        print("no changes")
    elif len(changed_coordinates) == 1:
        print("Single change detected")
        if determine_capture(current_characteristics, previous_characteristics, current_positions, previous_positions, previous_hist, changed_coordinates[0][0], changed_coordinates[0][1]):
            print("No bueno")
            skip_move = 1
    elif len(changed_coordinates) == 3:
        if detect_passant(current_positions, previous_positions, changed_coordinates):
            print("Fake Passant")
            skip_move = 1
    elif len(changed_coordinates) == 4:
        if detect_castle(current_positions, previous_positions, changed_coordinates):
            print("Fake Castle")
            skip_move = 1
    else:
        print("Multiple changes detected, unable to determine move.")
        skip_move = 1
    if skip_move == 0:
        print("We keeping dis one")
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
    print("Best move:", result.move)
    return result.move.from_square, result.move.to_square


def send_to_arduino(data): # 1 is right arm, 0 is left arm
    print("Sending to Arduino:", data)
    for value in data:
        arduino.write(bytes(f"{value},", 'utf-8'))

def determine_arm_to_use(square): # 1 is right arm, 0 is left arm
    file = square % 8
    if file <= 3:
        return 0
    else: 
        return 1
    
def exchange_pieces(from_square, from_arm, to_square, to_arm):
    print("Exchanging pieces:", from_square, from_arm, to_square, to_arm) #placeholder for later code

def determine_x_y(square_number):
    column = square_number % 8
    row = square_number // 8

    x1 = abs(column - 7) *square_size + default_x_offset + x_offsets[square_number] #default_x_offset shifts the whole grid cause arm is offset right, x_offsets shifts each column to better fit the square
    y1 = (row - 3) * square_size - square_size / 2 + y_offsets[square_number] #y_offsets shifts each row to better fit the square

    return x1, y1

def determine_servo_6(square_number):
    x, y = determine_x_y(square_number)
    angle = np.atan2(y, x)
    print("angle:", angle)
    return (map_range_clamped(angle, -np.pi/2, np.pi/2, -333, 333) + servo_baselines[5])

def determine_if_too_far(target_x, target_y, skip):
    global L1, L2
    if skip == True:
        return False
    distance = np.sqrt(target_x**2 + target_y**2)
    return False #distance > (L1 + L2) or distance < abs(L1 - L2) deal with this later it buggs out on bottom corner squares and returns true and breaks itself

def determine_wrist_tilt(servo_5_val, servo_4_val):
    global L1, L2, L3
    return max(min(129 - ((servo_5_val - servo_baselines[4]) + (servo_4_val - servo_baselines[3])), 1000), 0)

def determine_angles(target_x, target_y):
    print("target_x:", target_x, "target_y:", target_y)
    gamma = np.arctan2(target_y, target_x) 
    alpha = np.arccos(np.clip((target_x**2 + target_y**2 + L1**2 - L2**2) / (2 * L1 * np.sqrt(target_x**2 + target_y**2)), -1, 1))
    beta = np.arccos(np.clip((L1**2 + L2**2 - target_x**2 - target_y**2) / (2 * L1 * L2), -1, 1))

    Rtheta1 = gamma - alpha
    Ltheta1 = gamma + alpha
    Rtheta2 = np.pi - beta
    Ltheta2 = beta - np.pi

    #print("Rtheta1 (shoulder):", np.degrees(Rtheta1), "Ltheta1 (shoulder):", np.degrees(Ltheta1), "Rtheta2 (elbow):", np.degrees(Rtheta2), "Ltheta2 (elbow):", np.degrees(Ltheta2))

    zero_1 = servo_baselines[4]
    zero_2 = servo_baselines[3]

    zero_1_degrees = map_range_clamped(zero_1, 0, 1000, 0, 180)
    zero_2_degrees = map_range_clamped(zero_2, 0, 1000, 0, 270)

    Rdegree1 = (Rtheta1 * 180) / np.pi
    Ldegree1 = (Ltheta1 * 180) / np.pi
    Rdegree2 = (Rtheta2 * 180) / np.pi
    Ldegree2 = (Ltheta2 * 180) / np.pi

    # Everything 1 is shoulder, everything 2 is elbow

    RFinal1 = map_range_clamped(zero_1_degrees + Rdegree1, 0 , 180, 0, 1000)
    RFinal2 = map_range_clamped(zero_2_degrees + Rdegree2, 0, 180, 0, 1000)

    LFinal1 = map_range_clamped(zero_1_degrees + Ldegree1, 0 , 180, 0, 1000)
    LFinal2 = map_range_clamped(zero_2_degrees + Ldegree2, 0, 180, 0, 1000)
    #print("RFinal1 (shoulder):", RFinal1, "LFinal1 (shoulder):", LFinal1, "RFinal2 (elbow):", RFinal2, "LFinal2 (elbow):", LFinal2)
    return LFinal1, LFinal2, RFinal1, RFinal2

def determine_wrist_rotation(servo_6_val, direction):
    if direction == "horizontal":
        return servo_6_val
    if direction == "vertical":
        return 1000 - servo_6_val # I think this value is wrong; fix later

def move_to_highest_position(target_x, target_height, servo_6_val, square_num, type, arm): # this function puts the arm at the highest positoin to then be moved downwards
    global default_servo_positions
    max_target_height = target_height + 5
    LFinal1, LFinal2, RFinal1, RFinal2 = determine_angles(target_x, max_target_height)

    if type == "pickup":
        positions = []
        positions.append(default_servo_positions[0]) # gripper
        positions.append(determine_wrist_rotation(servo_6_val, "horizontal")) # wrist rotation
        positions.append(determine_wrist_tilt(LFinal1, LFinal2) + additional_wrist_offsets[square_num]) # wrist tilt
        positions.append(LFinal2)   # elbow
        positions.append(LFinal1)   # shoulder
        positions.append(servo_6_val) # base rotation
        positions.append(arm) # arm number
        send_to_arduino(positions)
        time.sleep(2.5)

    elif type == "dropoff":
        positions.append(1000) # gripper
        positions.append(determine_wrist_rotation(servo_6_val, "horizontal")) # wrist rotation
        positions.append(determine_wrist_tilt(LFinal1, LFinal2) + additional_wrist_offsets[square_num]) # wrist tilt
        positions.append(LFinal2)   # elbow
        positions.append(LFinal1)   # shoulder
        positions.append(servo_6_val) # base rotation
        positions.append(arm) # arm number
        send_to_arduino(positions)
        time.sleep(2.5)

def move_downwards(target_x, target_height, servo_6_val, square_num, type, arm): # this function goes from highest position down to lowest position
    global default_servo_positions
    new_height = target_height + 5

    
    for i in range(5):
        positions = []
        new_height -= 1

        LFinal1, LFinal2, RFinal1, RFinal2 = determine_angles(target_x, new_height)
        if type == "pickup":
            positions.append(default_servo_positions[0]) # gripper
            positions.append(determine_wrist_rotation(servo_6_val, "horizontal")) # wrist rotation
            positions.append(determine_wrist_tilt(LFinal1, LFinal2) + additional_wrist_offsets[square_num]) # wrist tilt
            positions.append(LFinal2)   # elbow
            positions.append(LFinal1)   # shoulder
            positions.append(servo_6_val) # base rotation
            positions.append(arm) # arm number
            send_to_arduino(positions)
            time.sleep(1.5)

        elif type == "dropoff":
            positions.append(1000) # gripper
            positions.append(determine_wrist_rotation(servo_6_val, "horizontal")) # wrist rotation
            positions.append(determine_wrist_tilt(LFinal1, LFinal2) + additional_wrist_offsets[square_num]) # wrist tilt
            positions.append(LFinal2)   # elbow
            positions.append(LFinal1)   # shoulder
            positions.append(servo_6_val) # base rotation
            positions.append(arm) # arm number
            send_to_arduino(positions)
            time.sleep(1.5)

    time.sleep(1.5)

    if type == "pickup":
        positions[0] = 1000 # closes gripper once down
        send_to_arduino(positions)
    elif type == "dropoff":
        positions[0] = default_servo_positions[0] # opens gripper once down
        send_to_arduino(positions)
    time.sleep(1.5)
    print("finished moving downwards, now sleeping for 2 seconds TRANSITION TRANSITION TRANSITION TRANSITION TRANSTITION TRANSITION TRANSITION TRANSITION TRANSITION TRANSITION")


def move_upwards(target_x, target_height, servo_6_val, square_num, type, arm): # this function goes from lowest position up to highest position
    global default_servo_positions
    new_height = target_height

    for i in range(5):
        positions = [] 
        new_height += 1

        LFinal1, LFinal2, RFinal1, RFinal2 = determine_angles(target_x, new_height)
        print("new_target_x:", target_x, "new_height:", new_height, "UPWARDS")
        if type == "pickup":
            positions.append(1000) # gripper
            positions.append(determine_wrist_rotation(servo_6_val, "horizontal")) # wrist rotation
            positions.append(determine_wrist_tilt(LFinal1, LFinal2) + additional_wrist_offsets[square_num]) # wrist tilt
            positions.append(LFinal2)   # elbow
            positions.append(LFinal1)   # shoulder
            positions.append(servo_6_val) # base rotation
            positions.append(arm) # arm number
            send_to_arduino(positions)
            time.sleep(1.5)
        elif type == "dropoff":
            positions.append(default_servo_positions[0]) # gripper
            positions.append(determine_wrist_rotation(servo_6_val, "horizontal")) # wrist rotation
            positions.append(determine_wrist_tilt(LFinal1, LFinal2) + additional_wrist_offsets[square_num]) # wrist tilt
            positions.append(LFinal2)   # elbow
            positions.append(LFinal1)   # shoulder
            positions.append(servo_6_val) # base rotation
            positions.append(arm) # arm number
            send_to_arduino(positions)
            time.sleep(1.5) #greater than 7, less than 8

def move_to_default(arm):
    global default_servo_positions
    positions = [
        default_servo_positions[0],
        default_servo_positions[1],
        default_servo_positions[2],
        default_servo_positions[3],
        default_servo_positions[4],
        default_servo_positions[5],
        arm
    ]
    send_to_arduino(positions) # 1 is right arm, 0 is left arm

def move_arm_to_square(square_num, type, arm): # This function starts the entire movement section
    target_height = target_y_list[square_num] # For some reason, it ends slightly higher than pawn height. Added -.25 to compensate, but this is a band-aid fix and should be looked into later
    target_x, target_y = determine_x_y(square_num)
    servo_6_val = determine_servo_6(square_num)

    move_to_highest_position(target_x, target_height, servo_6_val, square_num, type, arm)
    move_downwards(target_x, target_height, servo_6_val, square_num, type, arm)
    print("finished moving downwards, now moving upwards TRANSITION TRANSITION TRANSITION TRANSITION TRANSTITION TRANSITION TRANSITION TRANSITION TRANSITION TRANSITION")
    time.sleep(1.5)
    move_upwards(target_x, target_height, servo_6_val, square_num, type, arm)
    print("finished moving upwards, now moving to default TRANSITION TRANSITION TRANSITION TRANSITION TRANSTITION TRANSITION TRANSITION TRANSITION TRANSITION TRANSITION")



"""initialize_camera()
initial_characteristics = infer_chess_board([cv2.imread("C:\\Chess_Images\\WIN_20260303_18_29_00_Pro.jpg"),
                                             cv2.imread("C:\\Chess_Images\\WIN_20260303_18_29_01_Pro.jpg"),
                                             cv2.imread("C:\\Chess_Images\\WIN_20260303_18_29_02_Pro (2).jpg"),
                                             cv2.imread("C:\\Chess_Images\\WIN_20260303_18_29_02_Pro.jpg"),
                                             cv2.imread("C:\\Chess_Images\\WIN_20260303_18_29_03_Pro.jpg")
                                            ])

starting_values()

move_to_default(1)
time.sleep(.035)
move_to_default(0)

while True: 
    if check_for_move():
        f, t = determine_move()
        from_arm = determine_arm_to_use(f)
        to_arm = determine_arm_to_use(t)
        if from_arm != to_arm:
            exchange_pieces(f, from_arm, t, to_arm)
        else: # if the move is within the same arm, arm passed as parameter does not matter; can be to_arm or from_arm
            move_arm_to_square(f, "pickup", from_arm)
            time.sleep(1.5)
            move_arm_to_square(t, "dropoff", from_arm)
        move_to_default(from_arm)

"""
time.sleep(2) #This is just for testing cuz arduino needs time to link w computer
move_arm_to_square(37, "pickup", 1)
"""

while True:
    check_for_move()
"""


"""
original_image = take_pictures()[0]
warped_image = warp_board(original_image)
if original_image is None:
    print("Error: Could not capture image.")
cv2.imshow("Original", original_image)
cv2.imshow("Warped", warped_image)
cv2.waitKey(0)
print("dimensions:", original_image.shape)
print("Intensity: ")
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
print("4, 4: mean intensity:", cv2.mean(center[:, :, 2])[0], ", edges:", np.count_nonzero(cv2.Canny(blurred_image, 75, 225)) / blurred_image.size, ", variance:", np.var(blurred_image), ", Deviation:", np.std(blurred_image))

x_start = 5 * square_size
y_start = 6 * square_size
square = image[y_start + map_range(7, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(6, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
blurred_image = cv2.GaussianBlur(square, (5, 5), 0)
hsv = hsv_image[y_start + map_range(7, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(6, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
h, w, _ = hsv.shape
center = cv2.GaussianBlur(hsv[h//4:3*h//4, w//4:3*w//4], (3, 3), 0)
print("5, 6: mean intensity:", cv2.mean(center[:, :, 2])[0], ", edges:", np.count_nonzero(cv2.Canny(blurred_image, 75, 225)) / blurred_image.size, ", variance:", np.var(blurred_image), ", Deviation:", np.std(blurred_image))

x_start = 7 * square_size
y_start = 7 * square_size
square = image[y_start + map_range(8, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(8, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
blurred_image = cv27.GaussianBlur(square, (5, 5), 0)
hsv = hsv_image[y_start + map_range(8, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(8, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
h, w, _ = hsv.shape
center = cv2.GaussianBlur(hsv[h//4:3*h//4, w//4:3*w//4], (3, 3), 0)
print("7, 7: mean intensity:", cv2.mean(center[:, :, 2])[0], ", edges:", np.count_nonzero(cv2.Canny(blurred_image, 75, 225)) / blurred_image.size, ", variance:", np.var(blurred_image), ", Deviation:", np.std(blurred_image))

x_start = 0 * square_size
y_start = 0 * square_size
square = image[y_start + map_range(0, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(0, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
blurred_image = cv2.GaussianBlur(square, (5, 5), 0)
hsv = hsv_image[y_start + map_range(0, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(0, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
h, w, _ = hsv.shape
center = cv2.GaussianBlur(hsv[h//4:3*h//4, w//4:3*w//4], (3, 3), 0)
print("0, 0: mean intensity:", cv2.mean(center[:, :, 2])[0], ", edges:", np.count_nonzero(cv2.Canny(blurred_image, 75, 225)) / blurred_image.size, ", variance:", np.var(blurred_image), ", Deviation:", np.std(blurred_image))

x_start = 7 * square_size
y_start = 0 * square_size
square = image[y_start + map_range(0, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(8, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
blurred_image = cv2.GaussianBlur(square, (5, 5), 0)
hsv = hsv_image[y_start + map_range(0, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(8, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
h, w, _ = hsv.shape
center = cv2.GaussianBlur(hsv[h//4:3*h//4, w//4:3*w//4], (3, 3), 0)
print("7, 0: mean intensity:", cv2.mean(center[:, :, 2])[0], ", edges:", np.count_nonzero(cv2.Canny(blurred_image, 75, 225)) / blurred_image.size, ", variance:", np.var(blurred_image), ", Deviation:", np.std(blurred_image))

x_start = 0 * square_size
y_start = 7 * square_size
square = image[y_start + map_range(8, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(0, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
blurred_image = cv2.GaussianBlur(square, (5, 5), 0)
hsv = hsv_image[y_start + map_range(8, 0, 8, lower_mapy, upper_mapy):y_start + square_size, x_start + map_range(0, 0, 8, lower_mapx, upper_mapx):x_start + square_size]
h, w, _ = hsv.shape
center = cv2.GaussianBlur(hsv[h//4:3*h//4, w//4:3*w//4], (3, 3), 0)
print("0, 7: mean intensity:", cv2.mean(center[:, :, 2])[0], ", edges:", np.count_nonzero(cv2.Canny(blurred_image, 75, 225)) / blurred_image.size, ", variance:", np.var(blurred_image), ", Deviation:", np.std(blurred_image))



hsv = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
square = hsv[4 * square_size:4 * square_size + square_size, 4 * square_size:4 * square_size + square_size]
cv2.imshow("Square 4x4", square)
avg_v = cv2.mean(hsv[:, :, 2])[0]
print("4x4", avg_v)

square = hsv[5 * square_size:4 * square_size + square_size, 6 * square_size:4 * square_size + square_size]
cv2.imshow("Square 5x6", square)
cv2.waitKey(0)
avg_v = cv2.mean(hsv[:, :, 2])[0]
print("5x6", avg_v)


x_start = 4 * square_size
y_start = 4 * square_size
square = image[y_start:y_start + square_size, x_start:x_start + square_size]
mean_intensity = cv2.mean(square)[0]
print(mean_intensity)
print(", ")

x_start = 5 * square_size
y_start = 6 * square_size
square = image[y_start:y_start + square_size, x_start:x_start + square_size]
mean_intensity = cv2.mean(square)[0]
print(mean_intensity)

x_start = 4 * square_size
y_start = 4 * square_size
square = image[y_start:y_start + square_size - 2, x_start:x_start + square_size]
cv2.imshow("Square 4,4", square,)
cv2.waitKey(0)

edges = cv2.Canny(blurred_image, 75, 225)
print("4, 4: edges:", np.count_nonzero(edges) / edges.size) #, ", variance:", np.var(blurred_image), ", Deviation:", np.std(blurred_image))



x_start = 5 * square_size
y_start = 6 * square_size
square = image[y_start:y_start + square_size + 2, x_start + 1:x_start + square_size]
cv2.imshow("Square 5, 6", square)

blurred_image = cv2.GaussianBlur(square, (5, 5), 0)
edges = cv2.Canny(blurred_image, 75, 225)
cv2.imshow("Edges 5,6", edges)
cv2.waitKey(0)
print("6, 7: edges:", np.count_nonzero(edges) / edges.size) #, ", variance:", np.var(blurred_image), ", Deviation:", np.std(blurred_image))
"""
