# Control mini robot, based on face detection -- using Tensorflow Lite + Google TPU
# RKD - 2024.03.29

# TODO:
# * Make separate threads for movement and face/object detect so movement doesn't block detection
# * Get rid of the last of the dlib face-detect cruft
# X Finally got TPU installed on RPi5 using the script from here:
#   https://forums.raspberrypi.com/viewtopic.php?t=363962


import cv2
from picamera2 import Picamera2
import libcamera
#from simple_pid import PID
import math
import time
from RpiMotorLib import RpiMotorLib
import concurrent.futures
import Ultrasonic_distance as ud
import threading
import numpy as np
import tflite_runtime.interpreter as tflite


#from pprint import *

# Tensorflow lite setup / config
interpreter = tflite.Interpreter(model_path='coral/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite',
                                  experimental_delegates=[tflite.load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Other setup / config
#dlib_type = 0  # 0=original; 1 is faster dlib; 2 is more advanced one, but slow
image_scale = 1.0
image_dim = (640,480) # 1296, 927; (1280, 960)}  #1296,972 


nogui = False
debug = True
nomove = True

################ Face Timer #####

face_detected = False
face_timer = []

def clear_face_detected():
    global face_detected
    face_detected = False
    if debug: print("Clearing Face Detected")
    
def start_face_timer():
    global face_detected
    global face_timer
    face_detected=True
    if debug: print("Starting Face Detected Timer")
    try:
        face_timer.cancel()
    except:
        pass
    face_timer=threading.Timer(3, clear_face_detected)
    face_timer.start()


################### Motor stuff ##############

GpioPins_left = [18, 23, 24, 25]
GpioPins_right = [4, 5, 6, 13]

motor_left = RpiMotorLib.BYJMotor("MyMotorOne", "28BYJ")
motor_right = RpiMotorLib.BYJMotor("MyMotorTwo", "28BYJ")
motor_delay = 0.001
motor_mode = "half"
motor_steps_per_rev = 512   # half mode, 4096 steps per revolution????

def turn_left(steps=25, motor_slow_factor=1):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(motor_left.motor_run, GpioPins_left, motor_delay*motor_slow_factor, steps, False, False, motor_mode, .05)
        executor.submit(motor_right.motor_run, GpioPins_right, motor_delay*motor_slow_factor, steps, False, False, motor_mode, .05)

def turn_right(steps=25, motor_slow_factor=1):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(motor_left.motor_run, GpioPins_left, motor_delay*motor_slow_factor, steps, True, False, motor_mode, .05)
        executor.submit(motor_right.motor_run, GpioPins_right, motor_delay*motor_slow_factor, steps, True, False, motor_mode, .05)

def move_forward(steps=50, motor_slow_factor=2):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(motor_left.motor_run, GpioPins_left, motor_delay*motor_slow_factor, steps, False, False, motor_mode, .05)
        executor.submit(motor_right.motor_run, GpioPins_right, motor_delay*motor_slow_factor, steps, True, False, motor_mode, .05)
    
def move_backward(steps=50, motor_slow_factor=2):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(motor_left.motor_run, GpioPins_left, motor_delay*motor_slow_factor, steps, True, False, motor_mode, .05)
        executor.submit(motor_right.motor_run, GpioPins_right, motor_delay*motor_slow_factor, steps, False, False, motor_mode, .05)
 


######### Face Detect ################

def find_closest_face(faces, image_mid_x, image_mid_y):
    face_num = -1
    face_dist = 9999
    current_face = -1
    for rect in faces:
        current_face += 1
        if dlib_type == 0:
            (x,y,w,h) = faces[face_num]
            face_mid_x = x + w//2
            face_mid_y = y + h//2                   
        else:
            face_mid_x = (rect.left() + rect.right()) // 2
            face_mid_y = (rect.top() + rect.bottom()) // 2
        this_face_dist = math.sqrt((face_mid_x - image_mid_x)**2 +
                         (face_mid_y - image_mid_y)**2)
        if this_face_dist < face_dist:
            face_dist = int(this_face_dist)
            face_num = current_face
    return face_num
    
def check_for_faces(camera, detector):
    image = camera.capture_array()
    image = cv2.resize(image, dsize=None, fx=image_scale, fy=image_scale)  # was 0.5, 0.5  # there is a tradeoff between image size

    if dlib_type == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector(image_rgb, 0)
        faces = [r.rect for r in results]
    elif dlib_type == 1:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector(image_rgb, 0)
    else:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            image_gray,
            scaleFactor = 1.1,  #was 1.2
            minNeighbors = 8,
            minSize = (70,70)
            )
    return faces, image
    

########################### MAIN #######################
    
def run_robot():
    global face_detected
    
    last_direction = 0
        
    pc2 = Picamera2()
    #pprint(pc2.sensor_modes)
    
    preview_config = pc2.create_preview_configuration()
    preview_config['transform'] = libcamera.Transform(vflip=1, hflip=0)
    preview_config['main']={"format": 'RGB888', "size": image_dim} # 1296, 927; (1280, 960)}  #1296,972
    pc2.configure(preview_config)
    pc2.start(show_preview=False)
        
    scan_speed = 3
    scan_amount = 10
    num_search_steps = 20
    
    while True:  
        dist = ud.distance()  # Cliff detect

        image = pc2.capture_array()
        image = cv2.resize(image, (width, height) )
        input_data = np.expand_dims(image, axis=0)
        image_mid_x = image.shape[1]//2
        image_mid_y = image.shape[0]//2
        imH, imW = image.shape[:2]
        
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        
        if debug: print(f"Scores: {scores[:6]}")

        # Show bounding boxes
        for i in range(len(scores)):
            if ((scores[i] > 0.05) and (scores[i] <= 1.0)):
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))

                cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

        
        # Move
        if (scores[0] > 0.01): # Face Found
            i = 0
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            face_mid_x = (xmin+xmax) // 2
            face_mid_y = (ymax+ymin) // 2
            face_width = xmax - xmin
            
            FOV=75 # field of view of the camera
            diff_deg = (face_mid_x - image_mid_x)/image_mid_x * (FOV/2)
            steps = (motor_steps_per_rev/360)*diff_deg#* 1.5 # 1.2 fudge factor
            if debug: print(f"steps: {steps}")
            
            face_width_thresh = (200/240)*image_mid_x # was 150/240
            #if debug: print(f"face_width: {face_width}, threshold: {face_width_thresh}")
            
            ctrl = -steps
            
            min_ctrl = 5
            if nomove:
                pass
            elif ctrl > min_ctrl: # Turn to the left
                if debug: print("left")
                turn_left(abs(ctrl))
            elif ctrl < -min_ctrl:  # Turn to the right
                if debug: print("right")
                turn_right(abs(ctrl))
            elif face_width > face_width_thresh:
                move_backward(20,2)
                pass
            elif dist < 10.0 and face_width < 0.9*face_width_thresh: 
                move_forward(10,4)
            else:
                pass

        
        
        cv2.imshow('Object detector', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



if __name__ == "__main__":
    run_robot()

