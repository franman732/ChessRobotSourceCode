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


width = 750
height = 750
true_corners = np.float32([
    [637, 88], # Top left
    [1331, 94], # Top right
    [1311, 805], # Bottom right
    [635, 796], # Bottom left
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
edges_strength = 6.5
variance_strength = 9.5
deviation_strength = 150
threshold =  900
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))

upper_mapx = 17
lower_mapx = 10
upper_mapy = 9
lower_mapy = 5

check_1 = True
previous_positions = None
previous_characteristics = None

cap = None
hist_images = []
previous_hist = None

def map_range(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

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
        ret, frame = timed_call(cap.read)
        if ret:
            photos.append(frame)
        else:
            photos.append(None)
        print("bouttaloop5timeslul")
        for _ in range(5):
            timed_call(cap.read)
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
            inferred_board = np.array([[[0 for _ in range(8)] for _ in range(8)] for _ in range(6)], dtype=np.float32)
            warped_image = warp_board(starting_image)
            square_size = warped_image.shape[0] // 8

            gray_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
            hsv_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
            lab_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2LAB)
            blurred_for_variance = cv2.GaussianBlur(gray_image, (3, 3), 0)
            blurred_image = cv2.GaussianBlur(hsv_image, (3, 3), 0)
            blurred_for_lab = cv2.GaussianBlur(lab_image, (3, 3), 0)
            edge_image = clahe.apply(blurred_for_variance)

            for row in range(8):
                for col in range(8):
                    mapped_row = map_range(row, 0, 7, lower_mapy, upper_mapy)
                    mapped_col = map_range(col, 0, 7, lower_mapx, upper_mapx)

                    x_start = col * square_size
                    y_start = row * square_size
                    blurred_HSV_square = blurred_image[y_start + mapped_row + square_size // 5:(y_start + square_size) - square_size//5, x_start + mapped_col + square_size//5:(x_start + square_size) - square_size//5]
                    blurred_square = blurred_for_variance[y_start + mapped_row + square_size // 5:(y_start + square_size) - square_size//5, x_start + mapped_col + square_size//5:(x_start + square_size) - square_size//5]
                    lab_square = blurred_for_lab[y_start + mapped_row + square_size // 5:(y_start + square_size) - square_size//5, x_start + mapped_col + square_size//5:(x_start + square_size) - square_size//5]
                    edge_square = edge_image[y_start + mapped_row + square_size // 5:(y_start + square_size) - square_size//5, x_start + mapped_col + square_size//5:(x_start + square_size) - square_size//5]
                    edges = np.count_nonzero(cv2.Canny(edge_square, 100, 225))
                    variance = np.var(blurred_square)
                    deviation = np.std(blurred_square)
                    h = blurred_HSV_square[:,:,0]
                    s = blurred_HSV_square[:,:,1]
                    hist_h = cv2.calcHist([h], [0], None, [16], [0, 180])
                    hist_s = cv2.calcHist([s], [0], None, [16], [0, 256])
                    hist = np.concatenate([hist_h, hist_s])
                    hist = cv2.normalize(hist, hist).flatten()
                    L = np.mean(lab_square[:,:,0])
                    A = np.mean(lab_square[:,:,1], axis=(0,1))
                    B = np.mean(lab_square[:,:,2], axis=(0,1))
                    inferred_board[0][row][col] = edges
                    inferred_board[1][row][col] = variance
                    inferred_board[2][row][col] = deviation
                    inferred_board[3][row][col] = L
                    inferred_board[4][row][col] = A
                    inferred_board[5][row][col] = B
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
    global current_characteristics
    
    photos = take_pictures()
    current_characteristics = infer_chess_board(photos)

    positions = np.array([[['E' for _ in range(8)] for _ in range(8)] for _ in range(5)])
    diff = np.array([[[0 for _ in range(8)] for _ in range(8)] for _ in range(5)], dtype=np.int64)
    true_positions = np.array([['E' for _ in range(8)] for _ in range(8)])
    for i in range(5):
        for row in range(8):
            for col in range(8):
                edges_dif = np.float64(abs(current_characteristics[i][0][row][col] - initial_characteristics[i][0][row][col]))
                variance_dif = np.float64(abs(current_characteristics[i][1][row][col] - initial_characteristics[i][1][row][col]))
                deviation_dif = np.float64(abs(current_characteristics[i][2][row][col] - initial_characteristics[i][2][row][col]))
                total_dif = np.float64(edges_dif * edges_strength + variance_dif * variance_strength + deviation_dif * deviation_strength)
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
        from_square = chess.square(col, 7 - row)
    if from_square is None:
        return True
    for move in board.legal_moves:
        if move.from_square == from_square:
            if current_positions[(7 - chess.square_rank(move.to_square))][chess.square_file(move.to_square)] == 'X':
                row_2 = 7 - chess.square_rank(move.to_square)
                col_2 = chess.square_file(move.to_square)
                prev_hist = np.mean(array_hist_p[:, row_2, col_2, :], axis=0)
                curr_hist = np.mean(array_hist_n[:, row_2, col_2, :], axis=0)
                prev_L = np.mean([previous_characteristics[i][3][row_2][col_2] for i in range(5)])
                new_L = np.mean([current_characteristics[i][3][row_2][col_2] for i in range(5)])
                prev_A = np.mean([previous_characteristics[i][4][row_2][col_2] for i in range(5)])
                new_A = np.mean([current_characteristics[i][4][row_2][col_2] for i in range(5)])
                prev_B = np.mean([previous_characteristics[i][5][row_2][col_2] for i in range(5)])
                new_B = np.mean([current_characteristics[i][5][row_2][col_2] for i in range(5)])
                to_square = chess.square(col_2, 7 - row_2)
                move2 = chess.Move(from_square, to_square)
                if ((cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA) > .1) and ((abs(prev_L - new_L) > 50) or ((abs(prev_A - new_A) > 2) and (abs(prev_B - new_B) > 2)))):
                    piece = board.piece_at(from_square)
                    if piece and piece.piece_type == chess.PAWN and ((chess.square_rank(to_square) == 0) or (chess.square_rank(to_square) == 7)):
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
                        print("not legal")
                else:
                    print("Mean not strong enough for capture: ", )
                    print("Hist: ", cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA))
                    print("L: ", abs(prev_L - new_L), "A: ", abs(prev_A - new_A), "B: ", abs(prev_B - new_B))
                    print("move: ", move2)
    return True

def detect_castle(current_positions, previous_positions, changed_coordinates):
    king_row = king_col = rook_row = rook_col = None
    KING_ROW = KING_COL = ROOK_ROW = ROOK_COL = None
    king = 0
    rook = 0
    KING = 0
    ROOK = 0
    two_change = 0
    move = None
    if board.turn == chess.WHITE:
        turn = "white"
    else:
        turn = "black"
    for position in changed_coordinates:
        row = 7 - position[0]
        col = position[1]
        square = chess.square(col, row)
        piece = board.piece_at(square)
        if turn == "black":
            if ((previous_positions[position[0]][position[1]] == 'X' and current_positions[position[0]][position[1]] == 'E') and piece is not None and piece.piece_type == chess.KING):
                king = 1
                king_row = position[0]
                king_col = position[1]
            if ((previous_positions[position[0]][position[1]] == 'X' and current_positions[position[0]][position[1]] == 'E') and piece is not None and piece.piece_type == chess.ROOK):
                rook = 1
                rook_row = position[0]
                rook_col = position[1]
            if ((previous_positions[position[0]][position[1]] == 'E' and current_positions[position[0]][position[1]] == 'X') and board.piece_at(square) is None):
                two_change += 1
        if turn == "white":
            if ((previous_positions[position[0]][position[1]] == 'X' and current_positions[position[0]][position[1]] == 'E') and piece is not None and piece.piece_type == chess.KING):
                KING = 1
                KING_ROW = position[0]
                KING_COL = position[1]
            if ((previous_positions[position[0]][position[1]] == 'X' and current_positions[position[0]][position[1]] == 'E') and piece is not None and piece.piece_type == chess.ROOK):
                ROOK = 1
                ROOK_ROW = position[0]
                ROOK_COL = position[1]
            if ((previous_positions[position[0]][position[1]] == 'E' and current_positions[position[0]][position[1]] == 'X') and board.piece_at(square) is None):
                two_change += 1
    if king == 1 and rook == 1 and two_change == 2 and (king_row == 0 and rook_row == 0):
        if rook_col == 7 and king_col == 4:
            move = chess.Move.from_uci("e8g8")
        elif rook_col == 0 and king_col == 4:
            move = chess.Move.from_uci("e8c8")
    if KING == 1 and ROOK == 1 and two_change == 2 and (KING_ROW == 7 and ROOK_ROW == 7):
        if ROOK_COL == 7 and KING_COL == 4:
            move = chess.Move.from_uci("e1g1")
        elif ROOK_COL == 0 and KING_COL == 4:
            move = chess.Move.from_uci("e1c1")
    if move in board.legal_moves:
        board.push(move)
        print(board)
        print("Detected castle move:", move)
        return False
    else:
        print("Not legal castle: ", move, "two_change:", two_change, "king:", king, "rook:", rook, "KING:", KING, "ROOK:", ROOK)
        return True

def detect_passant(current_positions, previous_positions, changed_coordinates):
    from_square = to_square = None
    move = None
    move_1 = 0
    pawn_row_1 = pawn_col_1 = pawn_row_2 = pawn_col_2 = end_col = None
    for position in changed_coordinates:
        row = 7 - position[0]
        col = position[1]
        square = chess.square(col, row)
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
            from_square = chess.square(pawn_col_2, 7 - pawn_row_1)
        elif pawn_col_2 == end_col:
            from_square = chess.square(pawn_col_1, 7 - pawn_row_2)
    if from_square is not None:
        move = chess.Move(from_square, to_square)
    if move in board.legal_moves:
        board.push(move)
        print("Detected en passant move:", move)
        return False
    print("Not legal en passant:", move)
    return True

def check_for_move():
    global check_1, previous_positions, previous_characteristics, hist_images, previous_hist
    skip_move = 0
    changed_coordinates = []
    if check_1 == True:
        previous_positions = starting_values()
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
                to_square = chess.square(c, 7 - r)
            elif previous_positions[r][c] == 'X' and current_positions[r][c] == 'E':
                from_square = chess.square(c, 7 - r)
        if from_square is None or to_square is None:
            print("Could not determine from/to squares.")
            skip_move = 1
            return
        else:
            move = chess.Move(from_square, to_square)
        if move in board.legal_moves:
            piece = board.piece_at(from_square)
            print("Detected move:", move)
            if piece and piece.piece_type == chess.PAWN and ((chess.square_rank(to_square) == 0) or (chess.square_rank(to_square) == 7)):
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
            print("Illegal move detected!")
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


"""
def determine_move():
    info = engine.analyse(board, chess.engine.Limit(time=0.1))
    result = engine.play(board, chess.engine.Limit(time=1))
    print("Best move:", result.move)
    board.push(result.move)
"""
initialize_camera()
initial_characteristics = infer_chess_board([cv2.imread("C:\\Chess_Images\\WIN_20260109_20_58_57_Pro.jpg"),
                                             cv2.imread("C:\\Chess_Images\\WIN_20260109_20_58_58_Pro (2).jpg"),
                                             cv2.imread("C:\\Chess_Images\\WIN_20260109_20_58_58_Pro.jpg"),
                                             cv2.imread("C:\\Chess_Images\\WIN_20260109_20_59_00_Pro.jpg"),
                                             cv2.imread("C:\\Chess_Images\\WIN_20260109_20_59_01_Pro.jpg")
                                            ])


starting_values()
while True:
    check_for_move()







"""
def initialize_values():
    while not_initialized == True:
        button_clicked = check_button()
        if button_clicked == True:
            global initial_characteristics
            initial_characteristics = infer_chess_board("C:\\Chess_Images\\image_for_initial_dark.jpg")
            not_initialized = False



def determine_positions():
    current_characteristics = infer_chess_board(take_picture())
    positions = np.array([['E' for _ in range(8)] for _ in range(8)])
    diff = np.array([[0 for _ in range(8)] for _ in range(8)], dtype=np.int16)
    for row in range(8):
        for col in range(8):
            edges_dif = np.float64(abs(current_characteristics[0][row][col] - initial_characteristics[0][row][col]))
            variance_dif = np.float64(abs(current_characteristics[1][row][col] - initial_characteristics[1][row][col]))
            deviation_dif = np.float64(abs(current_characteristics[2][row][col] - initial_characteristics[2][row][col]))
            total_dif = np.float64(edges_dif * edges_strength + variance_dif * variance_strength + deviation_dif * deviation_strength)
            if total_dif > threshold:
                positions[row][col] = 'X'  
            else: 
                positions[row][col] = 'E'
            diff[row][col] = total_dif
    print("Differences:", diff)
    return positions






original_image = take_picture()
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
blurred_image = cv2.GaussianBlur(square, (5, 5), 0)
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
