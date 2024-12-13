# PhyzAI remote control, with face detection
# Alternate version with head-mounted camera
#
# Initial Rev, RKD 2024-08
#
# TODO:
# * Add space/location to face detection: if too far from current spot, dump the current name
# X Change to FaceNet instead of face_recognition library. The latter doesn't seem to work really well.
# X Add Face Recognition
# * Add back the random-faces if found-head-count is less than target head count
# X Remove camera-on-head (relative motion) code
# X add calibration offset to head
# XX Only look for heads when not moving??
# XX Debug: Keep track of head location in Phyz_control_space when camera is head-mounted.
#       This isn't working properly.  The motion of the head messes up face detection.
# X Add some random head moves, and looking around if no face is detected
# X Tweak calibration of Phyz head versus image center
# XX add second camera to Phyz's head?  Easier to track exact location of people (closed loop)
# X Make 1st camera wider angle of view.  S/W fix?
# X Tracked location (red box  of a real face) does not exactly match circle drawn (green) 
# X Pose changes should happen with a different cadence than changing people
# X Phyz should track a real persons slight movements
# X Add camera
# X Detect Faces
# X Move sort-of randomly to detected faces.



# Libraries to install
# - pip install pygame
# - pip install opencv-python
# - pip install numpy   # for the display
# - put zmaestro.py in same directory as this file

# Also install the Maestro driver and control
# - https://www.pololu.com/file/0J266/maestro-windows-130422.zip
# - run Maestro Control Center
# - in Serial Settings, choose "USB Dual Port"
# - and choose "Never Sleep"


# Info Links
# Search for Playstation 4 Controller here:
# https://www.pygame.org/docs/ref/joystick.html 
#
# Servo controller "Maestro Mini"
# https://www.pololu.com/product/1350/resources
# https://github.com/frc4564/maestro


# Enable different basic operations

HOME = True    # At Keith's house
enable_GUI = True
enable_MC = False # enable Motor Control
enable_face_detect = True
enable_show_phyz_loc = True
enable_randomize_look = False # Look around a little bit for each face
enable_face_camera = False # Look more straight ahead


num_people = 5   # *Maximum* number of "people" to include in the scene
FACE_DET_TTL = 30  # Hold-time for face detction interruptions (in ticks)
FACE_NAME_TTL = 30 # Hold-time for a named face (in ticks)

# Calibration to get the head to face you exactly (hopefully)
HEAD_OFFSET_X = 16
HEAD_OFFSET_Y = 0


import pygame
import cv2 
import numpy as np
import face_recognition
import time
#if enable_face_detect:
from facenet_pytorch import MTCNN
#import tensorflow as tf
##from tensorflow.keras.models import load_model
#from keras.models import load_model
#model_path = "./facenet_keras.h5"
#facenet_model = load_model(model_path)

from keras_facenet import FaceNet
embedder = FaceNet()

# FIXME: Choose correct com-port and device
if enable_MC:
    import zmaestro as maestro
    if HOME:
        servo = maestro.Controller('COM3', device=2)  # Keith @ home; Check COM port in Windows Device Manager
    else:
        servo = maestro.Controller('COM5', device=1)  # Phyz; Check COM port in Windows Device Manager



# Servo Definitions

head_x_channel = 1   # Determine what these actually are on Phyz's Maestro board with Maestro Control Center
head_y_channel = 0
head_tilt_channel = 2
arm_left_channel = 4
arm_right_channel = 3

# Get these from real PhyzAI Maestro board
head_x_range = (1520*4, 1620*4, 1728*4) # head left/right
head_y_range = (735*4, 936*4, 1136*4) # head up / down
#head_tilt_range = (1400*4, 1450*4, 1500*4)   # old.  Not sure why it changed. 
head_tilt_range = (1237*4, 1337*4, 1437*4) 
arm_right_range = (1536*4, 2608*4, 2608*4) 
arm_left_range = (1184*4, 1184*4, 1744*4)
        




##################
# Functions
##################

def draw_face_boxes(frame, boxes, probs):
        """ Draw a box for each face detected """
        if boxes is None:
            pass
        else:
            for box, prob in zip(boxes, probs): #, landmarks):
                cv2.rectangle(frame,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                (0, 0, 255),
                                thickness=2)
                # Show probability
                cv2.putText(frame, str(
                    round(prob,2)), (int(box[2]), int(box[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                #draw_person_loc(frame,int((box[0]+box[2])//2), int((box[1]+box[3])//2), (100,0,0))
                #print(box)
        return frame


def draw_person_loc(image, pos_x, pos_y, face_name = "unknown", color = (0, 200, 0)):
    """ Draw an oval where each face is located """
    axesLength = (20, 40) 
    startAngle = 0
    endAngle = 360
    thickness = 3
    angle = 0
    cv2.ellipse(image, (pos_x, pos_y), axesLength, 
           angle, startAngle, endAngle, color, thickness)
    cv2.putText(image, face_name, (pos_x-15, pos_y+65), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return image



def draw_phyz_position(image, pos_x, pos_y, angle=0, left_arm=0, right_arm=0, note=""): 
    """ Draw an image of the current head and arm positions (in screen-units, not ideal units) """

    # Ellipse for the head
    axesLength = (50, 100) 
    startAngle = 0
    endAngle = 360
    color = (255, 0, 0) 
    thickness = 5
    cv2.ellipse(image, (pos_x, pos_y), axesLength, 
           angle, startAngle, endAngle, color, thickness)
    
    # Ellipse of left arm
    #left_arm = max(left_arm, 0)
    axesLength = (10, max(int(50-40*left_arm),0)) 
    startAngle = 0
    endAngle = 360
    color = (255, 0, 0) 
    thickness = 4
    cv2.ellipse(image, (pos_x-70, pos_y+40-int(40*left_arm)), axesLength, 
           0, startAngle, endAngle, color, thickness)
    
    # Ellipse of right arm
    #right_arm = max(right_arm, 0)
    axesLength = (10, max(int(50-40*right_arm),0)) 
    startAngle = 0
    endAngle = 360
    color = (255, 0, 0) 
    thickness = 5
    cv2.ellipse(image, (pos_x+70, pos_y+40-int(40*right_arm)), axesLength, 
           0, startAngle, endAngle, color, thickness)
        
    cv2.putText(image, note, (pos_x-45, pos_y+130), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    return image
  

def choose_people_locations(num_people = 5):
    """return a list of people, where each pair shows percentage of total range available"""
    #people_list = np.random.randint(-90, 90, (num_people,2))
    people_list = []
    for i in range(num_people):
        #people_list = [[0,0]]
        this_x = np.random.randint(-80,80)
        this_x = np.random.randint(10,80) * np.random.choice((-1,1))
        this_y = np.random.randint(-25,25)
        people_list.append([this_x, this_y, []])
        
    return people_list
     

def get_screen_position(person_loc = [0,0], move_scale = 1.0):
    """ Translate Ideal person location to point on the screen """
    x_pos = person_loc[0]
    y_pos = person_loc[1]

    #x_scale = int(0.8*image_size_x / 2)
    #y_scale = image_size_y // 2 // 2  # don't look too much up or down

    #x_pos = int(image_size_x/2 + x_loc*x_scale/100)
    #y_pos = int(image_size_y/2 + y_loc*y_scale/100)

    x_box_mid = int(move_scale * image_size_x*(x_pos + 100)/200)
    y_box_mid = int(move_scale * image_size_y*(y_pos + 100)/200)
                
    return(x_box_mid,y_box_mid)


#################
# Motor Control 
#################

def set_head_to_nominal():
    servo.setTarget(head_x_channel, head_x_range[1])
    servo.setTarget(head_y_channel, head_y_range[1])
    servo.setTarget(head_tilt_channel, head_tilt_range[1])
    servo.setTarget(arm_left_channel, arm_left_range[1])
    servo.setTarget(arm_right_channel, arm_right_range[1])

    #FIXME: tweak these while watching the hw
    speed=80
    servo.setSpeed(head_x_channel, speed)
    servo.setSpeed(head_y_channel, speed//3)
    servo.setSpeed(head_tilt_channel, speed//4)
    servo.setSpeed(arm_left_channel, speed)
    servo.setSpeed(arm_right_channel, speed)

    accel = 2  #FIXME: tweak this
    servo.setAccel(head_x_channel, 1*accel)
    servo.setAccel(head_y_channel, 1*accel)
    servo.setAccel(head_tilt_channel, accel)
    servo.setAccel(arm_left_channel, accel)
    servo.setAccel(arm_right_channel, accel)


def move_physical_position(person_loc=[0,0], angle=0, left_arm=0, right_arm=0, move_relative=False, move_scale=1.0):
    """
    Translate Ideal person location and head/arms to physical position.
    * Ideal is based on a -100 to +100 in x and y dimensions.
    """

    # Get actual camera (head) position
    head_x = servo.getPosition(head_x_channel)  #FIXME: Convert this to Ideal-plane 
    head_x = (head_x - (head_x_range[0])) / (head_x_range[2]-head_x_range[0])
    head_x = head_x * 200 - 100
    head_y = servo.getPosition(head_y_channel)
    head_y = (head_y - (head_y_range[0])) / (head_y_range[2]-head_y_range[0])
    head_y = head_y * 200 - 100

    if move_relative:
        x_loc = person_loc[0] - head_x
        y_loc = person_loc[1] - head_y
    else:
        x_loc = person_loc[0] - HEAD_OFFSET_X
        y_loc = person_loc[1] - HEAD_OFFSET_Y

    x_scale = int(move_scale*(head_x_range[2] - head_x_range[0])/2 )
    y_scale = int(move_scale*(head_y_range[2] - head_y_range[0])/2 )
    x_pos = int(head_x_range[1] + x_loc*x_scale/100)   # FIXME: should head_x_range[1] be the subtraction 2 lines up???
    y_pos = int(head_y_range[1] + y_loc*y_scale/100)

    angle_scale = int((head_tilt_range[2] - head_tilt_range[0])/2)
    head_pos = int(head_tilt_range[1] + angle*angle_scale/45)

    arm_left_scale = (arm_left_range[2] - arm_left_range[0])
    arm_right_scale = (arm_right_range[2] - arm_right_range[0])
    arm_left_pos = int(arm_left_range[0] + left_arm*arm_left_scale)
    arm_right_pos = int(arm_right_range[0] + right_arm*arm_right_scale)

    servo.setTarget(head_x_channel, x_pos)
    servo.setTarget(head_y_channel, y_pos)
    servo.setTarget(head_tilt_channel, head_pos)
    servo.setTarget(arm_left_channel, arm_left_pos)
    servo.setTarget(arm_right_channel, arm_right_pos)

    return

import os
import glob
import re


def get_pos_from_box(box):
    x_box_mid = int((box[0]+box[2])/2)
    y_box_mid = int((box[1]+box[3])/2)
    x_pos = (x_box_mid/image_size_x) * 200 - 100
    y_pos = (y_box_mid/image_size_y) * 200 - 100
    return (x_pos, y_pos)


def generate_encodings_from_dir(directory):
    """Scan for jpg files in a directory.  Generate a list of encodings for Facial Recognition"""
    #face_encodings = []
    #face_names = []
    known_faces = []
    for file in os.listdir(directory):
        if file.endswith(".jpg"):
            file_path = os.path.join(directory, file)

            #target_image = face_recognition.load_image_file(file_path)
            target_image = cv2.imread(file_path)
            if target_image is not None:
                target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            else:
                continue

            #target = face_recognition.face_locations(target_image)
            this_face_encodings = embedder.extract(target_image, threshold=0.90) # face_region  
            if this_face_encodings:
                this_face_encoding = this_face_encodings[0]['embedding']
            # Use a regular expression to match the alphabetic part of the filename
                match = re.match(r"([a-zA-Z]+)", file)      
                if match:
                    known_faces.append((match.group(1), this_face_encoding))
                    #face_encodings.append(target_encoding)
                else:
                    assert False
    return known_faces

#known_faces = generate_encodings_from_dir("./KnownFaces/")
#print(kf)
#print("")

def calculate_distance(embedding1, embedding2):
    """
    Computes the Euclidean distance between two face embeddings.
    """
    return np.linalg.norm(embedding1 - embedding2)


def check_for_face(target_encoding, threshold, known_faces):
    for face_name, face_encoding in known_faces:
        dist = calculate_distance(target_encoding, face_encoding)
        print("face_name, dist: ", face_name, dist)
        if dist <= threshold:
            return face_name
    return ("")
        
def check_region_for_known_face(face_region, thresh, known_faces):
    this_face_encodings = embedder.extract(face_region, threshold=thresh) # face_region
    if this_face_encodings:
        this_face_encoding = this_face_encodings[0]['embedding']
        face_name = check_for_face(this_face_encoding, 0.8, known_faces)
        print(face_name)
        fn_ttl = FACE_NAME_TTL
    else:
        face_name = ""
        #fn_ttl = 0
    return face_name

import math
def calc_face_dist(face_a_x, face_a_y, face_b_x, face_b_y):
    return math.sqrt( math.pow( (face_a_x - face_b_x),2) + math.pow((face_a_y - face_b_y) ,2) )


def preprocess_face(image, target_size=(160, 160)):
    """
    Preprocess the face for FaceNet encoding.
    Resize, normalize, and expand dimensions.
    """
    face = cv2.resize(image, target_size)
    face = face.astype('float32') / 255.0  # Normalize pixel values
    face = np.expand_dims(face, axis=0)    # Add batch dimension
    return face





############
### MAIN ###
############


print("*** Starting ***")
pygame.init()

# Create detector    
if enable_face_detect:
    mtcnn = MTCNN()
    known_faces = generate_encodings_from_dir("./KnownFaces/")

#face_names, face_encodings = generate_encodings_from_dir("./KnownFaces/")
#face_name = face_names[0]
#target_encoding = face_encodings[0]

# Video Capture and display (only 1st 2 backends work on Win11?)
if HOME:
    cap = cv2.VideoCapture(0)  #FIXME: Home camera needs this, PhyzAI camera needs below.  Why???
else:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # CAP_MSMF, CAP_DSHOW, _FFMPEG, _GSTREAMER
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = cap.read()
image_size_x = frame.shape[1]
image_size_y = frame.shape[0]

clock = pygame.time.Clock()

# Initialize on-screen Phyz 
pos_x = image_size_x//2
pos_y = image_size_y//2
head_angle = 0
left_arm = 0         
right_arm = 0


# Set head servos to nominal
if enable_MC: set_head_to_nominal

# Initialize stuff
person_num = 0
people_list = []
time_to_live = 0   # Time that detected faces stay in case of nothing new detected
head_duration_count = 0
body_duration_count = 0
person_offset_x = 0
person_offset_y = 0

last_looked_away = False  # check if the last state was "look away", so you always come back to a person
current_face_name = ""
known_face_x = 9999
known_face_y = 9999


if num_people > 0:
    random_people_list = choose_people_locations(num_people) 
    if enable_face_camera:
        random_people_list[0] = [0, 0, []]
else:
    random_people_list = []



### Main Loop ###

while True:

    clock.tick(30)  # Frame Rate = 30 fps

    # Read the frame from the webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)


    ### Detect Faces and draw on frame ###
    
    if enable_face_detect:
        boxes, probs = mtcnn.detect(frame, landmarks=False)
        #draw_face_boxes(frame, boxes, probs) #, landmarks)
    else:
        boxes = None

    if (boxes is None) and (time_to_live > 0):  # No face detected, but just keep to old state
        time_to_live -= 1
    elif (boxes is None) and (time_to_live <= 0):
        people_list = []
        #person_num = 0
    else:
        time_to_live = FACE_DET_TTL  # number of frames to ignore if no people detected
        people_list = []
        #person_num = 0
        for box, prob in zip(boxes, probs): 
            if prob > 0.90:

                try:
                    # Look for known faces
                    face_region = frame[ int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    #face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

                    ycrcb = cv2.cvtColor(face_region, cv2.COLOR_BGR2YCrCb)

                    # Equalize the Y channel
                    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])

                    # Convert back to BGR
                    face_region = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

                    cv2.imshow('face_region', face_region) 
                    cv2.moveWindow("face_region", 40,30)

                    pos_x, pos_y = get_pos_from_box(box)
                except:
                    pos_x = 0
                    pos_y = 0
                    face_region = []
                people_list.append((pos_x, pos_y, face_region))                



    ### Choose select faces or do random if none found ###            

    if len(people_list) < num_people:
        delta = num_people - len(people_list)
        people_list.extend(random_people_list[:delta])
    elif len(people_list) > num_people:
        people_list = people_list[:num_people]
    # elif len(people_list) == 0:
    #     #people_list = [[0,0]]
    #     this_x = int(np.random.normal(0, 20)) # np.random.randint(-30,30)
    #     this_y = int(np.random.normal(0, 10)) #np.random.randint(-25,25)
    #     people_list = [[this_x, this_y, []]]
    #     person_num = 0
    #     time_to_live = FACE_DET_TTL
    #     #print("random person: ", this_x, this_y)

    for person_x, person_y, _ in people_list:
        this_x, this_y = get_screen_position((person_x, person_y))
        draw_person_loc(frame, this_x, this_y, "")

    # Look at one person, or switch
    if head_duration_count <= 0:
        # Make person 0 most likely
        event_prob = np.random.randint(0,100)
        if event_prob < 50:   # Look at the main (same) person
            person_num = 0
            person_offset_x = 0
            person_offset_y = 0
            head_duration_count = abs(int(np.random.normal(15,25)))+10  # num of frames to keep looking at this person
            last_looked_away = False
            phyz_note = ""
            current_face_name = ""
        elif event_prob < 60 and enable_randomize_look:  # Look away a little
            person_offset_x = np.random.randint(5,10) * np.random.choice((-1,1))
            person_offset_y = np.random.randint(-5,5)
            head_duration_count = abs(int(np.random.normal(5,10)))+5  # num of frames to keep looking at this person
            last_looked_away = True
            print("Look away: ", person_offset_x, person_offset_y)
            phyz_note = "Glance"
        else:   # Switch person
            person_num = np.random.randint(0,len(people_list))
            print ("Switching person: ", person_num)
            person_offset_x = 0
            person_offset_y = 0
            head_duration_count = abs(int(np.random.normal(15,25)))+10  # num of frames to keep looking at this person
            last_looked_away = False
            phyz_note = ""
            current_face_name = ""
    else:
        head_duration_count -= 1
        #face_name = check_region_for_known_face

    #print("Head Duration Count, person: ", head_duration_count, person_num)
    

    ### Tilt Head, move arms ###

    if body_duration_count <= 0:  # Time for a new position
        head_angle = int(np.random.normal(0, 10))
        if np.random.randint(0,100) < 3: # hands up
            arm_left_axis = 0
            arm_right_axis = 1
        else:
            arm_left_axis = abs((np.random.normal(0.4, 0.3)))
            arm_right_axis = abs((np.random.normal(0.1, 0.3)))
        body_duration_count = int(np.random.normal(2,6))  # num of frames to keep same position
    else:
        body_duration_count -= 1
    
    
    person_x, person_y, person_face_region = people_list[person_num]
    # only check for known_face of person actually being looked at
    if (current_face_name == "") and (len(person_face_region) > 0):
        face_name = check_region_for_known_face(person_face_region, 0.6, known_faces)
        if len(face_name) > 0:
            current_face_name = face_name
            print("Found a known face: ", face_name)
    this_x, this_y = get_screen_position((person_x, person_y))
    
    # FIXME: Did this get rid of "hanging on to know face too long" problem???
    #if calc_face_dist(this_x, this_y, known_face_x, known_face_y) > 10:
    #    current_face_name = ""
    #known_face_x = this_x
    #known_face_y = this_y

    draw_person_loc(frame, this_x, this_y, current_face_name, (0, 200, 200))
    

    person_x += person_offset_x
    person_y += person_offset_y
    pos_x, pos_y = get_screen_position((person_x, person_y))
    if enable_GUI:
        if enable_show_phyz_loc: draw_phyz_position(frame, pos_x, pos_y, head_angle, arm_left_axis, arm_right_axis, phyz_note)
        cv2.imshow('image', frame) 
    if enable_MC:
        move_physical_position((person_x, person_y), head_angle, arm_left_axis, arm_right_axis)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    events = pygame.event.get()
    

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
print("EXITING NOW")
exit()
