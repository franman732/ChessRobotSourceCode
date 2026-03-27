import time
import numpy as np
import xarm
arm = xarm.Controller('USB')
L1 = 10
L2 = 9.5
L3 = 20.5

servo_baselines_right = [365, 0, 500, 504, 130, 511]
servo_baselines_left = [365, -50, 500, 509, 134, 500]  #Index 1, actually counts as an offset here, not a baseline, but it is easier to just put it here than to add an extra variable for it
default_servo_positions = [350, 495, 0, 274, 879, 511]

x_offsets = [1, .5, -.5, 0, -2.5, 1, .5, 1, 
             -0.5, -1.5, -1.5, -4, -3.5, -1.35, -1.7, -.5, 
             -2, -2.5, -3, -4, -3.25, -3, -2.55, -1.9, 
             -2, -3.5, -3.5, -4, -2, -3.15, -2.9, -5, 
             -2.5, -2.9, -3.3, -3.75, -2.4, -3, -3, 0, 
             -2, -2.55, -3, -4, -2, -2.5, -2, -1.8, 
             -.5, -1.7, -1.75, -4, -2, -1, -.5, .25, 
             1.75, 1, -1, -4, 0, 1.5, 1.5, 2.5] 

y_offsets = [2.5, 1.25, .5, 0, -.5, -2.5, -2.5, -1.75, 
             0, -1, -1, -2, 0, -.25, .25, 0,
             -1, -1.25, -1.5, -1.5, 0, .5, 1.25, 1.65,
             -1, -1, -.85, -1.25, -1, 0, 1, 1,
             -0.5, -.5, -.75, -1, 0, 0, 0, 0,
             0, 0, -.5, -.75, 0, 0, 0, -.3,
             -1.25, -1, -1.5,    0, -.5, 0, .5, 1,
             -7, -4.5, -1.75, -.5, 0, 1.75, 2, 4]

target_y_list = [-5, -6, -7.5, 0, -6.5, -6.8, -5.3, -4.5,
                -4.5, -5.3, -7, -7, -6, -6, -4.75, -3.75,
               -4, -5, -6, -7.5, -6.5, -5.25, -4, -3.5,
                -4, -5, -6, -7, -6.5, -5, -4, -2.5,
                -4, -5, -6, -7.5, -6.5, -5, -4, 0,
                -4, -5, -6, -7.5, -6.5, -5.25, -4, -3.4,
                -4.75, -5.5, -6.5, -7.5, -6, -6, -5, -4,
                -5.5, -6.5, -7.25, -8, 0, -6, -5.5, -5.5]

additional_wrist_offsets = [0, 0, +4, 0, +40, 0, 0, 0, #4
                            0, 0, 0, +30, +25, 0, 0, 0,
                            0, 0, 0, +20, +12, 0, 0, 0,
                            0, 0, 0, +20, +12, 0, 0, 0,
                            0, 0, 0, -5, +10, 0, 0, 0,
                            0, 0, 0, +15, +30, 0, 0, 0,
                            0, 0, 0, +20, +40, 0, 0, 0,
                            0, 0, +20, +45, 0, +55, 0, 0] # 60

# Numbers to fix: 61

special_offsets = {4: -25, 28: -12, 44: -25, 52: -40, 61: -40, 60: -12} # Only for elbow tilts at extremes

default_x_offset = 9
square_size = 3.3

square_num = 35

def map_range_clamped(x, in_min, in_max, out_min, out_max):
    x = max(min(x, in_max), in_min)
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def hundredth_map_range(x, in_min, in_max, out_min, out_max):
    return round((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min, 4)


def determine_x_y(square_number, arm):
    column = square_number % 8
    row = square_number // 8

    if arm == "right":
        x1 = (7 - column) *square_size + default_x_offset + x_offsets[square_number] #default_x_offset shifts the whole grid cause arm is offset right, x_offsets shifts each column to better fit the square
        y1 = (row - 3) * square_size - square_size / 2 + y_offsets[square_number] #y_offsets shifts each row to better fit the square
    elif arm == "left":
        x1 = column * square_size + default_x_offset + square_size // 2 + x_offsets[square_number] #default_x_offset shifts the whole grid cause arm is offset right, x_offsets shifts each column to better fit the square
        y1 = (4 - row) * square_size - square_size / 2 + y_offsets[square_number] #y_offsets shifts each row to better fit the square

    return x1, y1

def determine_servo_6(square_number, arm, servo_baselines):
    x, y = determine_x_y(square_number, arm)
    angle = np.atan2(y, x)
    print("angle:", angle)
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
    print("Pre_clamp: Rdegree1 (shoulder):", Rdegree1 + zero_1_degrees, "Ldegree1 (shoulder):", Ldegree1 + zero_1_degrees, "Rdegree2 (elbow):", Rdegree2 + zero_2_degrees, "Ldegree2 (elbow):", Ldegree2 + zero_2_degrees)
    RFinal1 = map_range_clamped(zero_1_degrees + Rdegree1, 0 , 180, 0, 1000)
    RFinal2 = map_range_clamped(zero_2_degrees + Rdegree2, 0, 180, 0, 1000)

    LFinal1 = map_range_clamped(zero_1_degrees + Ldegree1, 0 , 180, 0, 1000)
    LFinal2 = map_range_clamped(zero_2_degrees + Ldegree2, 0, 180, 0, 1000)
    print("RFinal1 (shoulder):", RFinal1, "LFinal1 (shoulder):", LFinal1, "RFinal2 (elbow):", RFinal2, "LFinal2 (elbow):", LFinal2)
    return LFinal1, LFinal2, RFinal1, RFinal2

def determine_wrist_rotation(servo_6_val, direction):
    if direction == "horizontal":
        return servo_6_val
    if direction == "vertical":
        return 1000 - servo_6_val # I think this value is wrong; fix later

def move_to_highest_position(target_x, target_height, servo_6_val, square_num, type, servo_baselines): # this function puts the arm at the highest positoin to then be moved downwards
    global default_servo_positions
    max_target_height = target_height
    LFinal1, LFinal2, RFinal1, RFinal2 = determine_angles(target_x, max_target_height, servo_baselines)

    if type == "pickup":
        arm.setPosition([ #moves to adjusted position now
        [1, 1000], # Later, adjust this to rotate based on square surrounding target square; this will probably happen in actual program, not here Also set this back to default_servo_positions[0] once done testing LOOK LOOK LOOK LOOK LOOK LOOK LOOK LOOK
        [2, determine_wrist_rotation(servo_6_val, "horizontal") + servo_baselines[1]],
        [3, determine_wrist_tilt(LFinal1, LFinal2, servo_baselines) + additional_wrist_offsets[square_num]],
        [4, LFinal2],   # elbow
        [5, LFinal1],   # shoulder
        [6, servo_6_val]
        ])
        time.sleep(2.5)
    elif type == "dropoff":
        arm.setPosition([ #moves to adjusted position now
        [1, 1000], # Later, adjust this to rotate based on square surrounding target square; this will probably happen in actual program, not here
        [2, determine_wrist_rotation(servo_6_val, "horizontal") + servo_baselines[1]],
        [3, determine_wrist_tilt(LFinal1, LFinal2, servo_baselines) + additional_wrist_offsets[square_num]],
        [4, LFinal2],   # elbow
        [5, LFinal1],   # shoulder
        [6, servo_6_val]
        ])
        time.sleep(2.5)

def move_downwards(target_x, target_height, servo_6_val, square_num, type, servo_baselines): # this function goes from highest position down to lowest position
    global default_servo_positions

    if ((square_num % 8 == 0 or square_num % 8 == 7) and  0 < square_num // 8 < 7): # if the square is in the leftmost or rightmost column, we want to skip the "too far" check because it will be slightly out of reach but still possible to grab
        difference = 3
    else:
        difference = 5

    new_height = target_height + difference
    for i in range(5):
        new_height -= 1

        LFinal1, LFinal2, RFinal1, RFinal2 = determine_angles(target_x, new_height, servo_baselines)

        if type == "pickup":
            arm.setPosition([ #moves to adjusted position now
            [1, default_servo_positions[0]], # Later, adjust this to rotate based on square surrounding
            [2, determine_wrist_rotation(servo_6_val, "horizontal") + servo_baselines[1]],
            [3, determine_wrist_tilt(LFinal1, LFinal2, servo_baselines) + additional_wrist_offsets[square_num]],
            [4, LFinal2],   # elbow
            [5, LFinal1],   # shoulder
            [6, servo_6_val]
            ])
            time.sleep(1)

        elif type == "dropoff":
            arm.setPosition([ #moves to adjusted position now
            [1, 1000], # Later, adjust this to rotate based on square surrounding
            [2, determine_wrist_rotation(servo_6_val, "horizontal") + servo_baselines[1]],
            [3, determine_wrist_tilt(LFinal1, LFinal2, servo_baselines) + additional_wrist_offsets[square_num]],
            [4, LFinal2],   # elbow
            [5, LFinal1],   # shoulder
            [6, servo_6_val]
            ])
            time.sleep(1)
    print("finished moving downwards, now sleeping for 2 seconds TRANSITION TRANSITION TRANSITION TRANSITION TRANSTITION TRANSITION TRANSITION TRANSITION TRANSITION TRANSITION")
    time.sleep(2)

    if type == "pickup":
        arm.setPosition([ # closes gripper once down
            [1, 1000]])
    elif type == "dropoff":
        arm.setPosition([ # opens gripper once down
            [1, default_servo_positions[0]]])
    
def move_upwards(target_x, target_height, servo_6_val, square_num, type, servo_baselines): # this function goes from lowest position up to highest position
    global default_servo_positions
    new_height = target_height
    for i in range(5): 
        new_height += 1

        LFinal1, LFinal2, RFinal1, RFinal2 = determine_angles(target_x, new_height, servo_baselines)
        print("new_target_x:", target_x, "new_height:", new_height, "UPWARDS")
        if type == "pickup":
            arm.setPosition([ #moves to adjusted position now
            [1, 1000], # Later, adjust this to rotate based on square surrounding
            [2, determine_wrist_rotation(servo_6_val, "horizontal") + servo_baselines[1]],
            [3, determine_wrist_tilt(LFinal1, LFinal2, servo_baselines) + additional_wrist_offsets[square_num]],
            [4, LFinal2],   # elbow
            [5, LFinal1],   # shoulder
            [6, servo_6_val]
            ])
            time.sleep(1)
        elif type == "dropoff":
            arm.setPosition([ #moves to adjusted position now
            [1, default_servo_positions[0]], # Later, adjust this to rotate based on square surrounding
            [2, determine_wrist_rotation(servo_6_val, "horizontal") + servo_baselines[1]],
            [3, determine_wrist_tilt(LFinal1, LFinal2, servo_baselines) + additional_wrist_offsets[square_num]],
            [4, LFinal2],   # elbow
            [5, LFinal1],   # shoulder
            [6, servo_6_val]
            ])
            time.sleep(1)

def move_to_default():
    global default_servo_positions
    arm.setPosition([
        [1, default_servo_positions[0]],
        [2, default_servo_positions[1]],
        [3, default_servo_positions[2]],
        [4, default_servo_positions[3]],
        [5, default_servo_positions[4]],
        [6, default_servo_positions[5]]
    ])

def move_arm_to_square(square_num, type): # This function starts the entire movement section
    if square_num % 8 < 4:
        arm_side = "left"
        servo_baselines = servo_baselines_left
    else:
        arm_side = "right"
        servo_baselines = servo_baselines_right
    target_height = target_y_list[square_num]
    target_x, target_y = determine_x_y(square_num, arm_side)
    servo_6_val = determine_servo_6(square_num, arm_side, servo_baselines)


    move_to_highest_position(target_x, target_height, servo_6_val, square_num, type, servo_baselines)
    time.sleep(5) #just for testing, remove later 
    move_downwards(target_x, target_height, servo_6_val, square_num, type, servo_baselines)
    print("finished moving downwards, now moving upwards TRANSITION TRANSITION TRANSITION TRANSITION TRANSTITION TRANSITION TRANSITION TRANSITION TRANSITION TRANSITION")
    time.sleep(1.5)
    move_upwards(target_x, target_height, servo_6_val, square_num, type, servo_baselines)
    print("finished moving upwards, now moving to default TRANSITION TRANSITION TRANSITION TRANSITION TRANSTITION TRANSITION TRANSITION TRANSITION TRANSITION TRANSITION")

move_arm_to_square(square_num, "pickup")