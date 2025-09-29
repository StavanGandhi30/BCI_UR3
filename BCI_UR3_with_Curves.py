from ur3 import *
from cortex import *
import tkinter as tk
import time
import threading
from dotenv import load_dotenv
import os
import sys
import logging
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import random

sys.path.append("..")

ROBOT_HOST = "140.192.35.37"
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

xmin, xmax, ymin, ymax, z = 0.15, 0.50, -0.22, 0.22, -0.40
data = [z, (ymax+ymin)/2, (xmax+xmin)/2, 2.4, 1.8, -2.615]

setp.input_double_register_0 = 0
setp.input_double_register_1 = 0
setp.input_double_register_2 = 0
setp.input_double_register_3 = 0
setp.input_double_register_4 = 0
setp.input_double_register_5 = 0
watchdog.input_int_register_0 = 0

sensitivity = 0.005

# start data synchronization
if not con.send_start(): sys.exit()

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

def left_key(intensity):
    global start_time, last_command
    if ((time.time() - start_time) < 3): 
        if last_command[0]:
            if last_command[0] == "up":
                moveTopLeft(last_command)
            if last_command[0] == "down":
                moveBottomLeft(last_command)
            last_command = [None, None]
            start_time = time.time()
        else:
            last_command[0] = "left"
            last_command[1] = intensity
            moveLeft()
    else:
        moveLeft()
        start_time = time.time()

def right_key(intensity):
    global start_time, last_command
    if ((time.time() - start_time) < 3): 
        if last_command[0]:
            if last_command[0] == "up":
                moveTopRight(last_command)
            if last_command[0] == "down":
                moveBottomRight(last_command)
            last_command = [None, None]
            start_time = time.time()
        else:
            last_command[0] = "right"
            last_command[1] = intensity
            moveRight()
    else:
        moveRight()
        start_time = time.time()

def up_key(intensity):
    global start_time, last_command
    if ((time.time() - start_time) < 3): 
        if last_command[0]:
            if last_command[0] == "left":
                moveTopLeft(last_command)
            if last_command[0] == "right":
                moveTopRight(last_command)
            last_command = [None, None]
            start_time = time.time()
        else:
            last_command[0] = "up"
            last_command[1] = intensity
            moveUp()

    else:
        moveUp()
        start_time = time.time()

def down_key(intensity):
    global start_time, last_command
    if ((time.time() - start_time) < 3): 
        if last_command[0]:
            if last_command[0] == "left":
                moveBottomLeft(last_command)
            if last_command[0] == "right":
                moveBottomRight(last_command)
            last_command = [None, None]
            start_time = time.time()
        else:
            last_command[0] = "down"
            last_command[1] = intensity
            moveDown()
    else:
        moveDown()
        start_time = time.time()

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

    con.send_pause()
    con.disconnect()

class Subscribe():
    def __init__(self, app_client_id, app_client_secret, app_profile_name, actions, sensitivity, debug_mode=False, **kwargs):
        self.c = Cortex(client_id=app_client_id, client_secret=app_client_secret, debug_mode=debug_mode, **kwargs)
        self.c.bind(create_session_done=self.on_create_session_done)
        self.c.bind(new_data_labels=self.on_new_data_labels)
        self.c.bind(new_com_data=self.on_new_com_data)
        self.c.bind(inform_error=self.on_inform_error)
        self.app_profile_name = app_profile_name
        self.loaded_profile = False
        self.actions = actions
        self.sensitivity = sensitivity

    def start(self, streams, headsetId=''):
        self.streams = streams

        if headsetId != '':
            self.c.set_wanted_headset(headsetId)

        self.c.open()

    def sub(self, streams):
        self.c.sub_request(streams)

    def unsub(self, streams):
        self.c.unsub_request(streams)

    def on_new_data_labels(self, *args, **kwargs):
        pass

    def on_new_com_data(self, *args, **kwargs):
        if self.loaded_profile == False:
            self.c.set_wanted_profile(self.app_profile_name)
            self.c.setup_profile(self.app_profile_name, "load")
            self.loaded_profile = True
            return
        
        self.c.set_mental_command_active_action(self.actions)
        eeg_data = kwargs.get('data')

        action = eeg_data.get('action')
        power = float(eeg_data.get('power'))

        if (action == "left"):
            left_key(power)

        if (action == "right"):
            right_key(power)

        if (action == "lift"):
            up_key(power)
    
        if (action == "drop"):
            down_key(power)

        print([action, power])
 
    def on_create_session_done(self, *args, **kwargs):
        self.sub(self.streams)

    def on_inform_error(self, *args, **kwargs):
        if (self.debug_mode):
            error_data = kwargs.get('error_data')
            print(error_data)

actions = ['lift','drop','left','right']
start_time = time.time()
last_command = [None, None]

load_dotenv()

if __name__ == "__main__":

    eeg = Subscribe(os.getenv("client_id"), os.getenv("client_secret"), os.getenv("profile_name"), actions, sensitivity, debug_mode=False)
    eeg_thread = threading.Thread(target=lambda: eeg.start(['com']), daemon=True)
    eeg_thread.start()
    
    ur3e_run_loop()