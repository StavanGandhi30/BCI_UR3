import tkinter as tk
import sys
import logging
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import threading
import time
import random

sys.path.append("..")

ROBOT_HOST = "192.168.4.125"
ROBOT_PORT = 30004
config_filename = "./rtde/control_loop_configuration.xml"

logging.getLogger().setLevel(logging.INFO)

conf = rtde_config.ConfigFile(config_filename)
state_names, state_types = conf.get_recipe("state")
setp_names, setp_types = conf.get_recipe("setp")
watchdog_names, watchdog_types = conf.get_recipe("watchdog")

con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
con.connect()

# get controller version
con.get_controller_version()

# setup recipes
con.send_output_setup(state_names, state_types)
setp = con.send_input_setup(setp_names, setp_types)
watchdog = con.send_input_setup(watchdog_names, watchdog_types)

watchdog.input_int_register_0 = 0

setp.input_double_register_0 = 0
setp.input_double_register_1 = 0
setp.input_double_register_2 = 0
setp.input_double_register_3 = 0
setp.input_double_register_4 = 0
setp.input_double_register_5 = 0

xmin, xmax, ymin, ymax, z = 0.15, 0.50, -0.22, 0.22, -0.40

data = [z, (ymax+ymin)/2, (xmax+xmin)/2, 2.4, 1.8, -2.615]

sensitivity = 0.005

# start data synchronization
if not con.send_start(): sys.exit()

def ur3e_run_loop():
    move_completed = True
    while True:
        # receive the current state
        state = con.receive()

        if state is None:
            break

        # do something...
        if move_completed and state.output_int_register_0 == 1:
            move_completed = False
            for i in range(0, 6):
                setp.__dict__["input_double_register_%i" % i] = data[i]
            # print("New pose = " + str(data))
            con.send(setp)
            watchdog.input_int_register_0 = 1
        elif not move_completed and state.output_int_register_0 == 0:
            move_completed = True
            watchdog.input_int_register_0 = 0

        # kick watchdog
        con.send(watchdog)


def moveTopLeft(last_command):
    if (((data[1]-sensitivity) > ymin) and ((data[2]+sensitivity) < xmax)):
        if last_command[0] == "up":
            data[1], data[2] = data[1] - sensitivity, data[2] + sensitivity * last_command[1]
        if last_command[0] == "left":
            data[1], data[2] = data[1] - sensitivity * last_command[1], data[2] + sensitivity
        print("Going Top-Left")

def moveBottomLeft(last_command):
    if (((data[1]-sensitivity) > ymin) and ((data[2]-sensitivity) > xmin)):
        if last_command[0] == "down":
            data[1], data[2] = data[1] - sensitivity, data[2] - sensitivity * last_command[1]
        if last_command[0] == "left":
            data[1], data[2] = data[1] - sensitivity * last_command[1], data[2] - sensitivity
        print("Going Bottom-Left")
    
def moveTopRight(last_command):
    if (((data[1]+sensitivity) < ymax) and ((data[2]+sensitivity) < xmax)):
        if last_command[0] == "up":
            data[1], data[2] = data[1] + sensitivity, data[2] + sensitivity * last_command[1]
        if last_command[0] == "right":
            data[1], data[2] = data[1] + sensitivity * last_command[1], data[2] + sensitivity
        print("Going Top-Right")

def moveBottomRight(last_command):
    if (((data[1]+sensitivity) < ymax) and ((data[2]-sensitivity) > xmin)):
        if last_command[0] == "down":
            data[1], data[2] = data[1] + sensitivity, data[2] - sensitivity * last_command[1]
        if last_command[0] == "right":
            data[1], data[2] = data[1] + sensitivity * last_command[1], data[2] - sensitivity
        print("Going Bottom-Right")

def moveRight():
    if (data[1]+sensitivity) < ymax:
        data[1]+=sensitivity
        print("Going Right")
        
def moveLeft():
    if (data[1]-sensitivity) > ymin:
        data[1]-=sensitivity
        print("Going Left")

def moveUp():
    if (data[1]+sensitivity) < ymax:
        data[2]+=sensitivity
        print("Going Up")

def moveDown():
    if ((data[2]-sensitivity) > xmin):
        data[2]-=sensitivity
        print("Going Down")

def handle_key(direction):
    global first_command_time, last_commands

    now = time.time()
    rand_val = random.uniform(0.4, 1)

    if not first_command_time:
        # Start a new combo window
        last_commands = {direction: rand_val}
        first_command_time = now
        return

    # If direction already pressed, update only if new value is higher
    if direction in last_commands:
        if rand_val > last_commands[direction]:
            last_commands[direction] = rand_val
    else: 
        # Add new direction
        last_commands[direction] = rand_val

    # Combo of two distinct directions is ready
    if len(last_commands) == 2:
        print("Making diagonals movement")

        dirs = sorted(last_commands.keys())
        key = tuple(dirs)

        movement_map = {
            ("down", "left"): moveBottomLeft,
            ("left", "down"): moveBottomLeft,
            ("down", "right"): moveBottomRight,
            ("right", "down"): moveBottomRight,
            ("up", "left"): moveTopLeft,
            ("left", "up"): moveTopLeft,
            ("up", "right"): moveTopRight,
            ("right", "up"): moveTopRight,
        }

        if key in movement_map:
            # Pass the direction with the highest weight
            max_dir = max(last_commands.items(), key=lambda x: x[1])
            movement_map[key](max_dir)
        else:
            print(f"No movement defined for combo: {key}")

        # Reset combo state
        last_commands = {}
        first_command_time = None
    
    elif len(last_commands) == 1 and (now - first_command_time > 2):
        print("Making single movement")

        direction = list(last_commands.keys())[0]
        single_move_map = {
            "left": moveLeft,
            "right": moveRight,
            "up": moveUp,
            "down": moveDown,
        }

        if direction in single_move_map:
            single_move_map[direction]()
        else:
            print(f"No single movement defined for: {direction}")

        # Reset after executing single direction
        last_commands = {}
        first_command_time = None


if __name__ == "__main__":
    last_commands = {}
    first_command_time = None

    ur3e_thread = threading.Thread(target=lambda: ur3e_run_loop(), daemon=True)
    ur3e_thread.start()

    main = tk.Tk()
    frame = tk.Frame(main, width=100, height=100)

    main.bind('<Left>', lambda e: handle_key("left"))
    main.bind('<Right>', lambda e: handle_key("right"))
    main.bind('<Up>', lambda e: handle_key("up"))
    main.bind('<Down>', lambda e: handle_key("down"))

    frame.pack()
    main.mainloop()
    con.send_pause()
    con.disconnect()
