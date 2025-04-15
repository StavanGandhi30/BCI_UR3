# importing pyautogui
import pyautogui
import time

# time.sleep()
pyautogui.click()

distance = 200

while distance:
    # moves the cursor to the right, specifying 'left' mouse button
    pyautogui.dragRel(distance, 0, duration=0.2, button='left')
    distance -= 10
	# move the cursor down 
    pyautogui.dragRel(0, distance, duration=0.2, button='left') 
	# move the cursor to the left 
    pyautogui.dragRel(-distance, 0, duration=0.2, button='left') 
    distance = distance - 5
	# move the cursor up 
    pyautogui.dragRel(0, -distance, duration=0.2, button='left') 
