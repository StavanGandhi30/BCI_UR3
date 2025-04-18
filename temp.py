import tkinter as tk
import sys
import logging
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import threading

sys.path.append("..")

ROBOT_HOST = "140.192.35.5"
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
if not con.send_start():
    sys.exit()

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
            print("New pose = " + str(data))
            con.send(setp)
            watchdog.input_int_register_0 = 1
        elif not move_completed and state.output_int_register_0 == 0:
            move_completed = True
            watchdog.input_int_register_0 = 0

        # kick watchdog
        con.send(watchdog)

def left_key(e):
    if ((data[1]-sensitivity) > ymin):
        data[1]-=sensitivity
    print("Left key pressed")

def right_key(e):
    if ((data[1]+sensitivity) < ymax):
        data[1]+=sensitivity
    print("Right key pressed")

def up_key(e):
    if ((data[2]+sensitivity) < xmax):
        data[2]+=sensitivity
    print("Up key pressed")

def down_key(e):
    if ((data[2]-sensitivity) > xmin):
        data[2]-=sensitivity
    print("Down key pressed")


if __name__ == "__main__":
    ur3e_thread = threading.Thread(target=lambda: ur3e_run_loop(), daemon=True)
    ur3e_thread.start()

    main = tk.Tk()
    frame = tk.Frame(main, width=100, height=100)

    main.bind('<Left>', left_key)
    main.bind('<Right>', right_key)
    main.bind('<Up>', up_key)
    main.bind('<Down>', down_key)

    frame.pack()
    main.mainloop()
    con.send_pause()
    con.disconnect()
