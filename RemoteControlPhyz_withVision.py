# PhyzAI remote control, with face detection
# Alternate version with head-mounted camera
#
# Initial Rev, RKD 2024-08
#
# TODO:
# * Change to FaceNet instead of face_recognition library. The latter doesn't seem to work really well.
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
enable_MC = False  # enable Motor Control
enable_face_detect = True


num_people = 3   # *Maximum* number of "people" to include in the scene
FACE_DET_TTL = 25  # Hold-time for face detction interruptions (in ticks)

# Calibration to get the head to face you exactly (hopefully)
HEAD_OFFSET_X = 16
HEAD_OFFSET_Y = 0


import pygame
import cv2 
import numpy as np
import face_recognition
import time
if enable_face_detect:
    from facenet_pytorch import MTCNN
    import tensorflow as tf
    from tensorflow.keras.models import load_model


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


def draw_person_loc(image, pos_x, pos_y, face_name = "unknown", color = (0, 100, 0)):
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
    people_list = np.random.randint(-90, 90, (num_people,2))
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
    servo.setAccel(head_x_channel, 5*accel)
    servo.setAccel(head_y_channel, 2*accel)
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
    face_encodings = []
    face_names = []
    for file in os.listdir(directory):
        if file.endswith(".jpg"):
            file_path = os.path.join(directory, file)


            # Set up "Keith" recognition
            #target_image_path = "Images/Keith100.jpg" # was 105
            target_image = face_recognition.load_image_file(file_path)
            target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            target = face_recognition.face_locations(target_image)
            #target_face = target_image[target[0][0]:target[0][2],
            #                        target[0][3]:target[0][1]]
            target_encoding = face_recognition.face_encodings(target_image)[0]

            # Use a regular expression to match the alphabetic part of the filename
            match = re.match(r"([a-zA-Z]+)", file)      
            if match:
                face_names.append(match.group(1))
                face_encodings.append(target_encoding)
            else:
                assert False
    return (face_names, face_encodings)

#face_names, face_encodings = generate_encodings_from_dir("./KnownFaces/")
#print(face_names)

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

# Set up "Keith" recognition
# target_image_path = "Images/Keith100.jpg" # was 105
# target_image = face_recognition.load_image_file(target_image_path)
# target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
# target = face_recognition.face_locations(target_image)
# target_face = target_image[target[0][0]:target[0][2],
#                            target[0][3]:target[0][1]]
# target_encoding = face_recognition.face_encodings(target_image)[0]
face_names, face_encodings = generate_encodings_from_dir("./KnownFaces/")
face_name = face_names[0]
target_encoding = face_encodings[0]

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
        person_num = 0
    else:
        time_to_live = FACE_DET_TTL  # number of frames to ignore if no people detected
        people_list = []
        person_num = 0
        for box, prob in zip(boxes, probs): 
            if prob > 0.70:
                # Look for known faces
                face_region = frame[ int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                cv2.imshow('face_region', face_region) 
                cv2.moveWindow("face_region", 40,30)
                this_face_encodings = face_recognition.face_encodings(face_region)
                if this_face_encodings:
                    found_faces = face_recognition.compare_faces(face_encodings, this_face_encodings[0])
                    print("Found face in region")
                    this_face_index = np.where(found_faces)
                    if this_face_index:
                        print("Found face index:", this_face_index[0][0])
                        face_name = face_names[this_face_index[0][0]]
                else:
                    face_name = "unknown"

                pos_x, pos_y = get_pos_from_box(box)
                people_list.append((pos_x, pos_y, face_name))
    
                # if encodings:
                #     #print("face encoded")
                #     dist = face_recognition.face_distance(encodings, target_encoding)
                #     print("dist: ", dist)
                #     if dist[0] < 0.6:
                #         #print("  Keith!")
                #         frame = cv2.putText(frame, face_name, (int(box[2]), int(box[3]+35)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                



    ### Choose select faces or do random if none found ###            

    if len(people_list) > num_people:
        people_list = people_list[:num_people]
    elif len(people_list) == 0:
        #people_list = [[0,0]]
        this_x = int(np.random.normal(0, 20)) # np.random.randint(-30,30)
        this_y = int(np.random.normal(0, 10)) #np.random.randint(-25,25)
        people_list = [[this_x, this_y, "random"]]
        person_num = 0
        time_to_live = FACE_DET_TTL
        #print("random person: ", this_x, this_y)

    for person_x, person_y, person_name in people_list:
        this_x, this_y = get_screen_position((person_x, person_y))
        draw_person_loc(frame, this_x, this_y, person_name)

    # Look at one person, or switch
    if head_duration_count <= 0:
        # Make person 0 most likely
        event_prob = np.random.randint(0,100)
        if event_prob < 20 or last_looked_away:   # Look at the main (same) person
            #person_num = 0
            person_offset_x = 0
            person_offset_y = 0
            head_duration_count = int(np.random.normal(5,7)+5)  # num of frames to keep looking at this person
            last_looked_away = False
            phyz_note = ""
        elif event_prob < 80:  # Look away a little
            person_offset_x = np.random.randint(10,30) * np.random.choice((-1,1))
            person_offset_y = np.random.randint(-20,20)
            head_duration_count = int(np.random.normal(2,5)+0)  # num of frames to keep looking at this person
            last_looked_away = True
            #print("Look away: ", person_offset_x, person_offset_y)
            phyz_note = "Glance"
        else:   # Switch person
            #print ("Switching person")
            person_num = np.random.randint(0,len(people_list))
            person_offset_x = 0
            person_offset_y = 0
            head_duration_count = int(np.random.normal(5,7)+5)  # num of frames to keep looking at this person
            last_looked_away = False
            phyz_note = ""
    else:
        head_duration_count -= 1


    ### Tilt Head, move arms ###

    if body_duration_count <= 0:  # Time for a new position
        head_angle = int(np.random.normal(0, 17))
        if np.random.randint(0,100) < 3: # hands up
            arm_left_axis = 0
            arm_right_axis = 1
        else:
            arm_left_axis = abs((np.random.normal(0.4, 0.3)))
            arm_right_axis = abs((np.random.normal(0.1, 0.3)))
        body_duration_count = int(np.random.normal(2,6))  # num of frames to keep same position
    else:
        body_duration_count -= 1
        

    # try:
    #     this_x, this_y = people_list[person_num]  # FIXME: Why does this get out of bounds????
    # except:
    #     this_x, this_y = (0,0)
    #     person_num = 0
    #pos_x, pos_y = get_screen_position(people_list[person_num])


    person_x, person_y, person_name = people_list[person_num]
    person_x += person_offset_x
    person_y += person_offset_y
    pos_x, pos_y = get_screen_position((person_x, person_y))
    if enable_GUI:
        draw_phyz_position(frame, pos_x, pos_y, head_angle, arm_left_axis, arm_right_axis, phyz_note)
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
