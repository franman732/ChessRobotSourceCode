import chess
import chess.engine
import time

import cv2
from matplotlib import image
import numpy as np

def timed_call(fn, *args, name=None):
    start = time.perf_counter()
    result = fn(*args)
    end = time.perf_counter()
    print(f"{name or fn.__name__} took {(end - start) * 1000:.2f} ms")
    return result

engine = chess.engine.SimpleEngine.popen_uci("C:\\Stockfish\\stockfish-windows-x86-64-avx2")
board = chess.Board()

BOARD_FLIPPED = True
board.turn = chess.BLACK

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

servo_3 = [686, 744, 765, 736, 770, 766, 707, 673,
            630, 706, 694, 763, 773, 786, 637, 642,
            620, 602, 756, 732, 766, 691, 678, 563,
            620, 668, 735, 765, 705, 670, 577, 574,
            580, 620, 727, 752, 732, 667, 598, 587,
            576, 683, 685, 791, 813, 662, 687, 571,
            609, 700, 755, 780, 744, 705, 713, 705,
            731, 766, 797, 787, 740, 800, 696, 667]
servo_4 = [206, 305, 381, 392, 821, 729, 606, 530,
            114, 234, 260, 430, 772, 732, 495, 446,
            89, 103, 336, 362, 735, 577, 517, 348,
            59, 162, 284, 399, 654, 553, 411, 348,
            27, 102, 273, 362, 689, 532, 420, 364,
            29, 169, 234, 462, 827, 548, 523, 381,
            97, 223, 361, 454, 697, 625, 583, 523,
            271, 363, 439, 496, 768, 778, 603, 528]
servo_5 = [727, 763, 803, 814, 139, 192, 252, 282,
            690, 734, 749, 831, 167, 194, 300, 326,
            681, 681, 788, 804, 197, 273, 292, 360,
            651, 706, 760, 821, 224, 276, 328, 366,
            641, 671, 750, 793, 216, 294, 328, 363,
            638, 693, 739, 850, 146, 279, 300, 330,
            677, 727, 800, 840, 219, 247, 273, 300,
            757, 804, 826, 860, 169, 178, 257, 290]
servo_6 = [690, 661, 640, 621, 369, 351, 331, 295,
            652, 630, 605, 586, 403, 383, 370, 342,
            606, 582, 563, 551, 440, 436, 422, 408,
            527, 524, 521, 513, 479, 482, 483, 479,
            457, 457, 473, 469, 516, 528, 536, 558,
            386, 400, 422, 429, 555, 577, 587, 625,
            349, 358, 378, 394, 564, 616, 642, 673,
            283, 317, 338, 374, 637, 647, 678, 713]
                
starting_servos = [593, 542, 1000, 679, 875, 493]
alignment_servos = []
servo_baselines = [365, 500, 458, 132, 866, 0]
move_time = 2500
additional_time_offset = 500

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
    global check_1, previous_positions, previous_characteristics, hist_images, previous_hist, initial_characteristics
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
        previous_hist = hist_images
"""    if previous_board != board:
        initial_characteristics = current_characteristics
        return True""" #comment this


def determine_move():
    engine.analyse(board, chess.engine.Limit(time=0.1))
    result = engine.play(board, chess.engine.Limit(time=1))
    print("Best move:", result.move)
    board.push(result.move)
    return result.move.from_square, result.move.to_square

def determine_arm_to_use(square):
    file = square % 8
    if file <= 3:
        return "Left"
    else: 
        return "Right"
    
def determine_wrist_rotation(servo_6_val, direction):
    if direction == "horizontal":
        return servo_6_val
    if direction == "vertical":
        return 1000 - servo_6_val # I think this value is wrong; fix later
    
def determine_wrist_tilt(servo_5_val, servo_4_val):
    return servo_baselines[2] + (servo_5_val - servo_baselines[4]) + (servo_4_val - servo_baselines[3])

def exchange_pieces(from_square, from_arm, to_square, to_arm):
    pass #placeholder for later code

def return_to_home(arm):
    arm.setPosition([[1, starting_servos[0]], [2, starting_servos[1]], [3, starting_servos[2]], [4, starting_servos[3]], [5, starting_servos[4]], [6, starting_servos[5]]], time=move_time)
    time.sleep((move_time + additional_time_offset)//1000)

def align_arm(arm):
    arm.setPosition([[1, alignment_servos[0]], [2, alignment_servos[1]], [3, alignment_servos[2]], [4, alignment_servos[3]], [5, alignment_servos[4]], [6, alignment_servos[5]]], time=move_time)
    time.sleep((move_time + additional_time_offset)//1000)

def simple_arm_move(from_square, to_square, arm):
    align_arm(arm)
    wrist_tilt = determine_wrist_tilt(servo_5[from_square], servo_4[from_square])

    arm.setPosition([[1, servo_baselines[0]],[2, servo_baselines[1]],[3, wrist_tilt],[4,servo_4[from_square]],[5,servo_5[from_square]],[6,servo_6[from_square]]], time=move_time)
    time.sleep((move_time + additional_time_offset)//1000)

    arm.setPosition


    


initialize_camera()
initial_characteristics = infer_chess_board([cv2.imread("C:\\Chess_Images\\WIN_20260206_17_24_19_Pro.jpg"),
                                             cv2.imread("C:\\Chess_Images\\WIN_20260206_17_24_20_Pro.jpg"),
                                             cv2.imread("C:\\Chess_Images\\WIN_20260206_17_24_21_Pro (2).jpg"),
                                             cv2.imread("C:\\Chess_Images\\WIN_20260206_17_24_21_Pro.jpg"),
                                             cv2.imread("C:\\Chess_Images\\WIN_20260206_17_24_22_Pro.jpg")
                                            ])

starting_values()
while True:
    if check_for_move():
        f, t = determine_move()
        from_arm = determine_arm_to_use(f)
        to_arm = determine_arm_to_use(t)
        if from_arm != to_arm:
            exchange_pieces(f, from_arm, t, to_arm)
        else:
            simple_arm_move(f, t, from_arm)

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
