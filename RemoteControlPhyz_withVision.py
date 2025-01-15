# PhyzAI remote control, with face detection
# Alternate version with head-mounted camera
#
# Initial Rev, RKD 2024-08
#
# TODO:
# * Maybe use YOLO for face (well, person) detection as well???
# * Face and ball TTL seem to not be working properly.  Generally works, but seems to switch too soon sometimes.
# X Face (and ball) time-to-live needs to be refactored.  Put face detect into its own function.  Wrap TTL around that.
# * Add space/location to face detection: if too far from current spot, dump the current name
# X Change to FaceNet instead of face_recognition library. The latter doesn't seem to work really well.
# X Add Face Recognition
# X Add back the random-faces if found-head-count is less than target head count
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

HOME = True   # At Keith's house
enable_GUI = True
enable_MC = False # enable Motor Control
enable_face_detect = True
enable_face_recog = True
enable_ball_detect=True
enable_show_phyz_loc = True
enable_randomize_look = False # Look around a little bit for each face
enable_face_camera = False # Look more straight ahead

likelihood_of_first_face = 50 # percent

num_people = 5   # *Maximum* number of "people" to include in the scene
FACE_DET_TTL = 20  # Hold-time for face detection (in ticks)

# Calibration to get the head to face you exactly (hopefully)
HEAD_OFFSET_X = 16
HEAD_OFFSET_Y = 0


import pygame
import cv2 
import numpy as np
import face_recognition
import time
from facenet_pytorch import MTCNN

import os
import glob
import re
import math


# Basic YOLO object detection
import torch
import cv2
from ultralytics import YOLO  # Use YOLO from the Ultralytics library
# Load a pre-trained YOLO model
# You can specify "yolov5s.pt" or "yolov8s.pt" (small model versions) or other model sizes for different performance
#model = YOLO("yolov8n.pt")  # 'n' for nano model, fast and lightweight for real-time detection
model = YOLO("yolo11s.pt")  # 'n' for nano model, fast and lightweight for real-time detection
#model = YOLO("/Volumes/Safari/PhyzAI_RemoteControl/runs/detect/train2/weights/best.pt")


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
arm_right_range = (896*4, 2608*4, 2608*4) 
arm_left_range = (944*4, 2000*4, 2000*4)
        



class Person:
    def __init__(self, x_pos = 0, y_pos = 0, face_region = [], name="", time_to_live=0):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.face_region = face_region
        self.name = name
        self.time_to_live = time_to_live



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
  

def choose_people_locations(num_people = 5, force_zero =  False):
    """return a list of people, where each pair shows percentage of total range available"""
    people_list = []
    for i in range(num_people):
        this_x = np.random.randint(-80,80)
        this_x = np.random.randint(10,80) * np.random.choice((-1,1))
        this_y = np.random.randint(-25,25)
        new_person = Person(this_x, this_y, [], "", 0)
        people_list.append(new_person)
    if force_zero:
        people_list[0] = Person(0, 0, [], "", 0)
        
    return people_list
     

def get_screen_position(person_loc = [0,0], move_scale = 1.0):
    """ Translate Ideal person location to point on the screen """
    x_pos = person_loc[0]
    y_pos = person_loc[1]

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


##################
# More Functions #
##################

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
        #fn_ttl = FACE_NAME_TTL
    else:
        face_name = ""
        #fn_ttl = 0
    return face_name


def calc_face_dist(face_a_x, face_a_y, face_b_x, face_b_y):
    return math.sqrt( math.pow( (face_a_x - face_b_x),2) + math.pow((face_a_y - face_b_y) ,2) )

def calc_person_face_dist(person1, person2):
    return calc_face_dist(person1.x_pos, person1.y_pos, person2.x_pos, person2.y_pos)

def preprocess_face(image, target_size=(160, 160)):
    """
    Preprocess the face for FaceNet encoding.
    Resize, normalize, and expand dimensions.
    """
    face = cv2.resize(image, target_size)
    face = face.astype('float32') / 255.0  # Normalize pixel values
    face = np.expand_dims(face, axis=0)    # Add batch dimension
    return face


# Ball (microphone) detection
def detect_ball(frame):
    items = ['apple', 'sports ball',] # 'person']  # FIXME: you can limit what YOLO actually looks for
    results = model(frame, verbose=False)
    #results = model.predict(frame, show=False, boxes=False, classes=False, conf=False)

    for result in results:
        boxes = result.boxes  # Each detection result has 'boxes', 'conf', and 'class' attributes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # Get coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = box.cls[0].item()  # Class ID

            this_item = model.names[int(cls)]
            this_item_conf = conf

            label = f"{model.names[int(cls)]} {conf:.2f}"
            if this_item in items:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                ball_x, ball_y = get_pos_from_box([x1,y1,x2,y2])
                return( ball_x, ball_y )  
            
    else:
        return(None, None)


def detect_faces(frame, max_faces = 5, PROB_THRESH = 0.90):
    face_list = []
    boxes, probs = mtcnn.detect(frame, landmarks=False)

    if boxes is not None:
        face_count = 0
        for box, prob in zip(boxes, probs): 
            if face_count >= max_faces:
                break
            if prob > PROB_THRESH:
                try:
                    # Set up region to look for known faces
                    face_region = frame[ int(box[1]):int(box[3]), int(box[0]):int(box[2])]

                    ycrcb = cv2.cvtColor(face_region, cv2.COLOR_BGR2YCrCb)

                    # Equalize the Y channel
                    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])

                    # Convert back to BGR
                    face_region = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

                    cv2.imshow('face_region', face_region) 
                    cv2.moveWindow("face_region", 40,30)

                    pos_x, pos_y = get_pos_from_box(box)
                    face_list.append(Person(pos_x, pos_y, face_region))    
                    face_count += 1            
                except:
                    pass
    return face_list
                



############
### MAIN ###
############


print("*** Starting ***")
pygame.init()

# Create detector    
if enable_face_detect:
    mtcnn = MTCNN()
    known_faces = generate_encodings_from_dir("./KnownFaces/")

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
if enable_MC: set_head_to_nominal()


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
        random_people_list[0] = Person(0, 0)  # Force 1st person to be face-on
else:
    random_people_list = []


people_list = random_people_list


### Main Loop ###

while True:

    clock.tick(30)  # Frame Rate = 30 fps

    # Read the frame from the webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if enable_face_detect:
        new_people_list = detect_faces(frame, num_people)

    if enable_ball_detect: # Ball / microphone detect
        ball_loc_x, ball_loc_y = detect_ball(frame)
        if ball_loc_x is not None:
            new_people_list.insert(0,Person(ball_loc_x, ball_loc_y, name="mic"))


    # Decrement time-to-live for each person
    for i in range(len(people_list)):
        if people_list[i].time_to_live > 0: 
            people_list[i].time_to_live = people_list[i].time_to_live - 1
        else:
            people_list[i].name = ""


    # Choose to keep a face or not, as they time-out
    for i in range(len(new_people_list)):
        # Check distance from new people to existing people
        new_face_dist = calc_person_face_dist(people_list[i], new_people_list[i])
        print("New Face Dist:", new_face_dist)

        if new_face_dist > 7:  # FIXME: what should this be???  Also, if a face is moving, should Phyz focus there?
            people_list[i] = new_people_list[i]
            people_list[i].time_to_live = FACE_DET_TTL
            people_list[i].name = ""
        elif (new_face_dist <= 5) and (people_list[i].time_to_live > 0):
            people_list[i].x_pos = new_people_list[i].x_pos
            people_list[i].y_pos = new_people_list[i].y_pos
        else:
            people_list[i] = random_people_list[i]
 

    
    # Recognize known faces

    if enable_face_recog:
        for person in people_list:
            if (person.name == "") and (len(person.face_region) > 0):
                face_name = check_region_for_known_face(person.face_region, 0.6, known_faces)
                if len(face_name) > 0:
                    person.name = face_name


    # Draw all the people

    for person in people_list:
        this_x, this_y = get_screen_position((person.x_pos, person.y_pos))
        draw_person_loc(frame, this_x, this_y, person.name)


    # Look at one person, or switch

    if head_duration_count <= 0:
        event_prob = np.random.randint(0,100)
        if event_prob < likelihood_of_first_face:   # Look at the main (same) person
            person_num = 0
            person_offset_x = 0
            person_offset_y = 0
            head_duration_count = abs(int(np.random.normal(1,20)))+7  # num of frames to keep looking at this person
            last_looked_away = False
            phyz_note = ""
            current_face_name = ""
        elif event_prob < 75 and enable_randomize_look:  # Look away a little
            person_offset_x = np.random.randint(5,10) * np.random.choice((-1,1))
            person_offset_y = np.random.randint(-10,10)
            head_duration_count = abs(int(np.random.normal(3,5)))+2  # num of frames to keep looking at this person
            last_looked_away = True
            print("Look away: ", person_offset_x, person_offset_y)
            phyz_note = "Glance"
        else:   # Switch person
            new_person_num = np.random.randint(0,len(people_list))
            if new_person_num == person_num:
                new_person_num = np.random.randint(0,len(people_list))
            person_num = new_person_num
            print ("Switching person: ", person_num)
            person_offset_x = 0
            person_offset_y = 0
            head_duration_count = abs(int(np.random.normal(3,15)))+5  # num of frames to keep looking at this person
            last_looked_away = False
            phyz_note = ""
            current_face_name = ""
    else:
        head_duration_count -= 1


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
    

    # Draw and move where Phyz is looking

    person_x = people_list[person_num].x_pos + person_offset_x
    person_y = people_list[person_num].y_pos + person_offset_y
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
