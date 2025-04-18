import socket
import math
import struct
import time

class UR3e:
    # Create a socket
    def __init__(self, host='140.192.35.5', port=30002):
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.connect((host, port))
            print(f"Connected to {host}:{port}")
        except Exception as e:
            print(f"Error: {e}")

    def moveLinear(self, x, y, z, rx, ry, rz, a=0.5, v=0.2):
        move_command = f"movel(p[{x}, {y}, {z}, {rx}, {ry}, {rz}], a={a}, v={v})\n"
        return self.send_command(move_command)

    # Send the command
    def send_command(self, command):
        try:
            self.s.sendall(command.encode('utf-8'))
            print(f"Command sent: {command}")
        except Exception as e:
            print(f"Error: {e}")
            return False
        return True

    # Close the socket
    def closeConnection(self):
        try:
            self.s.close()
            print("Connection closed.")
        except Exception as e:
            print(f"Error: {e}")
            return False
        return True

    def returnToHome(self, a=1, v=0.5):
        # moveToHome = f"movej([0, -1.5707963267948966, 0, -1.5707963267948966, 0, 0], a={a}, v={v})\n"
        moveToHome = f"movej([-1.5707963267948966, -1.5707963267948966, 0, -1.5707963267948966, 0, 0], a={a}, v={v})\n"
        return self.send_command(moveToHome)
        
    def moveToRandomPos(self, a=1, v=0.5):
        moveToHome = f"movej([1, -1.5707963267948966, 0, -1.5707963267948966, 0, 0], a={a}, v={v})\n"
        return self.send_command(moveToHome)

    def __angleToRadians(self, angle):
        return angle*(math.pi/180)

    def normalizeAngle(self, angle):
        radians = self.__angleToRadians(angle)
        isNeg = False

        if radians < 0:
            isNeg = True
            radians *= -1
        
        while (radians > (2*math.pi)):
            radians -= 2*math.pi

        return (radians * -1 if isNeg else radians)
    
    def getCurrentPose(self):
        try:
            # Create a new socket for reading robot state
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as state_socket:
                state_socket.connect((self.s.getpeername()[0], 30003))  # Use same IP, port 30003
                time.sleep(0.1)  # Wait a moment for data to accumulate

                state_data = state_socket.recv(1108)  # Primary interface returns 1108 bytes

                # Pose starts at byte 444 (index 444 to 491), 6 doubles (8 bytes each)
                pose_data = struct.unpack('!6d', state_data[444:444 + 48])
                
                print("Current TCP pose (x, y, z, rx, ry, rz):", pose_data)
                return pose_data
        except Exception as e:
            print(f"Error retrieving pose: {e}")
            return None
        
if __name__ == "__main__":
    robot = UR3e()
    time.sleep(2)
    robot.returnToHome()
    time.sleep(4)

    pose = robot.getCurrentPose()
    if pose:
        print(f"X: {pose[0]:.4f}, Y: {pose[1]:.4f}, Z: {pose[2]:.4f}\n")
        
    robot.moveLinear(-0.22499, 0.001, 0.29331, 2.431, 2.420, -2.401, a=0.1, v=0.1)

    time.sleep(2)
    robot.closeConnection()