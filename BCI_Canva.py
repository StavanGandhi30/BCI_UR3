from cortex import *
import tkinter as tk
import time
import pyautogui
import threading
from dotenv import load_dotenv
import os

load_dotenv()

class Subscribe():
    def __init__(self, app_client_id, app_client_secret, app_profile_name, actions, painter, sensitivity, debug_mode=False, **kwargs):
        self.c = Cortex(client_id=app_client_id, client_secret=app_client_secret, debug_mode=debug_mode, **kwargs)
        self.c.bind(create_session_done=self.on_create_session_done)
        self.c.bind(new_data_labels=self.on_new_data_labels)
        self.c.bind(new_com_data=self.on_new_com_data)
        self.c.bind(inform_error=self.on_inform_error)
        self.app_profile_name = app_profile_name
        self.loaded_profile = False
        self.actions = actions
        self.painter = painter
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
        data = kwargs.get('data')

        action = data.get('action')
        power = data.get('power')

        if (action == "left"):
            self.painter.moveLeft()
            time.sleep(0.1)
    
        if (action == "right"):
            self.painter.moveRight()
            time.sleep(0.1)

        if (action == "lift"):
            self.painter.moveUp()
            time.sleep(0.1)
    
        if (action == "drop"):
            self.painter.moveDown()
            time.sleep(0.1)

        print([action, power])
    
    def on_create_session_done(self, *args, **kwargs):
        self.sub(self.streams)

    def on_inform_error(self, *args, **kwargs):
        if (self.debug_mode):
            error_data = kwargs.get('error_data')
            print(error_data)

class Painter():
    def __init__(self, root, sensitivity):
        self.root = root
        self.root.attributes('-fullscreen', True)
        self.root.title("Paint App")
        self.sensitivity = sensitivity
        self.screenWidth, self.screenHeight = pyautogui.size()

        self.old_x = None
        self.old_y = None

        self.createCanvas()
        self.cursorInit()
        self.initKeys()

    def initKeys(self):
        self.canvas.focus_set()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        self.canvas.bind("<Left>", self.moveLeft)
        self.canvas.bind("<Right>", self.moveRight)

    def cursorInit(self):
        if (self.old_x == None or self.old_y == None):

            self.old_x = self.screenWidth/2
            self.old_y = self.screenHeight/2

    def createCanvas(self):
        self.canvas = tk.Canvas(self.root, width=500, height=500, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def moveLeft(self):
        self.cursorInit()

        new_x = self.old_x - self.sensitivity

        if (new_x >= 0):
            self.canvas.create_line(
                self.old_x, self.old_y, new_x, self.old_y,
                width=2, fill="green", capstyle=tk.ROUND, smooth=tk.TRUE
            )

            self.old_x = new_x

    def moveRight(self):
        self.cursorInit()

        new_x = self.old_x + self.sensitivity

        if (new_x <= self.screenWidth):
            self.canvas.create_line(
                self.old_x, self.old_y, new_x, self.old_y,
                width=2, fill="red", capstyle=tk.ROUND, smooth=tk.TRUE
            )

            self.old_x = new_x

    def moveUp(self):
        self.cursorInit()

        new_y = self.old_y + self.sensitivity

        if (new_y <= self.screenHeight):
            self.canvas.create_line(
                self.old_x, self.old_y, self.old_x, new_y,
                width=2, fill="green", capstyle=tk.ROUND, smooth=tk.TRUE
            )

            self.old_y = new_y

    def moveDown(self):
        self.cursorInit()

        new_y = self.old_y - self.sensitivity

        if (new_y >= 0):
            self.canvas.create_line(
                self.old_x, self.old_y, self.old_x, new_y,
                width=2, fill="green", capstyle=tk.ROUND, smooth=tk.TRUE
            )

            self.old_y = new_y
            
    def paint(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(
                self.old_x, self.old_y, event.x, event.y,
                width=2, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE
            )
        
        self.old_x, self.old_y = event.x, event.y

    def reset(self, event):
        self.old_x = None
        self.old_y = None

actions = ['lift','drop','left','right']
sensitivity = 10

if __name__ == "__main__":
    root = tk.Tk()
    app = Painter(root, sensitivity)
    eeg = Subscribe(os.getenv("client_id"), os.getenv("client_secret"), os.getenv("profile_name"), actions, app, sensitivity, debug_mode=False)

    # Start EEG stream in background
    eeg_thread = threading.Thread(target=lambda: eeg.start(['com']), daemon=True)
    eeg_thread.start()

    # Run Tkinter in main thread
    root.mainloop()