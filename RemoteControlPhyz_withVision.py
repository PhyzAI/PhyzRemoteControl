# PhyzAI remote control, with face detection
#
# Initial Rev, RKD 2024-08
#
# TODO:
# * Add camera
# * Detect Faces
# * Move sort-of randomly to detected faces.

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


import pygame
#import zmaestro as maestro
import cv2 
import numpy as np
#import tensorflow as tf
from facenet_pytorch import MTCNN


# Image Definitions

enable_GUI = True
enable_MC = False   # enable Motor Control
enable_face_detect = True

#image_size_x = 640
#image_size_y = 480

num_people = 2
FACE_DET_TTL = 15

# Servo Definitions

head_x_channel = 1   # Determine what these actually are on Phyz's Maestro board with Maestro Control Center
head_y_channel = 0
head_tilt_channel = 2
arm_left_channel = 4
arm_right_channel = 3

head_x_range = (1520*4, 1620*4, 1728*4)  # Get these from real PhyzAI Maestro board
head_y_range = (735*4, 936*4, 1136*4)
head_tilt_range = (1400*4, 1450*4, 1500*4) 
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
                # Draw rectangle on frame
                #print(box)
                cv2.rectangle(frame,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                (0, 0, 255),
                                thickness=2)

                # Show probability
                cv2.putText(frame, str(
                    round(prob,2)), (int(box[2]), int(box[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return frame


def draw_person_loc(image, pos_x, pos_y, color = (0, 100, 0)):
    """ Draw an oval where each (random) face is located """
    axesLength = (20, 40) 
    startAngle = 0
    endAngle = 360
    #color = (0, 100, 0) 
    thickness = 3
    angle = 0
    cv2.ellipse(image, (pos_x, pos_y), axesLength, 
           angle, startAngle, endAngle, color, thickness)
    return image



def draw_pos(image, pos_x,pos_y, angle=0, left_arm=0, right_arm=0): 
    """ Draw an image of the current head and arm positions """

    # Getting the height and width of the image 
    height = image.shape[0] 
    width = image.shape[1] 
    
    # Ellipse for the head
    axesLength = (50, 100) 
    startAngle = 0
    endAngle = 360
    color = (255, 0, 0) 
    thickness = 5
    cv2.ellipse(image, (pos_x, pos_y), axesLength, 
           angle, startAngle, endAngle, color, thickness)
    
    # Ellipse of left arm
    left_arm = max(left_arm, 0)
    axesLength = (10, int(50-40*left_arm)) 
    startAngle = 0
    endAngle = 360
    color = (255, 0, 0) 
    thickness = 5
    cv2.ellipse(image, (pos_x-70, pos_y+40-int(40*left_arm)), axesLength, 
           0, startAngle, endAngle, color, thickness)
    
    # Ellipse of right arm
    right_arm = max(right_arm, 0)
    axesLength = (10, int(50-40*right_arm)) 
    startAngle = 0
    endAngle = 360
    color = (255, 0, 0) 
    thickness = 5
    cv2.ellipse(image, (pos_x+70, pos_y+40-int(40*right_arm)), axesLength, 
           0, startAngle, endAngle, color, thickness)
        
    return image
  

def choose_people_locations(num_people = 5):
    # return a list of people, where each pair shows percentage of total range available
    people_list = np.random.randint(-100, 100, (num_people,2))
    people_list[0] = [0,0]
    return people_list
     
def get_position(person_loc = [0,0]):
    """ Translate relative person location to point on the screen """
    x_loc = person_loc[0]
    y_loc = person_loc[1]

    x_scale = int(0.8*image_size_x / 2)
    y_scale = image_size_y // 2 // 2  # don't look too much up or down

    x_pos = int(image_size_x/2 + x_loc*x_scale/100)
    y_pos = int(image_size_y/2 + y_loc*y_scale/100)
    return(x_pos,y_pos)

def move_physical_position(person_loc=[0,0], angle=0, left_arm=0, right_arm=0):
    """ Translate relative person location and head/arms to physical position """
    x_loc = person_loc[0]
    y_loc = person_loc[1]
    x_scale = int(0.8*(head_x_range[2] - head_x_range[0])/2 )
    y_scale = (head_y_range[2] - head_y_range[0])//2 // 2  # don't look too much up or down
    x_pos = int(head_x_range[1] + x_loc*x_scale/100)
    y_pos = int(head_y_range[1] + y_loc*y_scale/100)

    angle_scale = (head_tilt_range[2] - head_tilt_range[0])//2
    head_pos = int(head_tilt_range[1] + angle*angle_scale/45)

    arm_left_scale = (arm_left_range[2] - arm_left_range[0])
    arm_right_scale = (arm_right_range[2] - arm_right_range[0])
    arm_left_pos = int(arm_left_range[0] + left_arm*arm_left_scale)
    arm_right_pos = int(arm_right_range[0] + right_arm*arm_right_scale)

    if enable_MC:
        servo.setTarget(head_x_channel, x_pos)
        servo.setTarget(head_y_channel, y_pos)
        servo.setTarget(head_tilt_channel, head_pos)
        servo.setTarget(arm_left_channel, arm_left_pos)
        servo.setTarget(arm_right_channel, arm_right_pos)

    return




############
### MAIN ###
############

pygame.init()

# Create detector    
mtcnn = MTCNN()

# Video Capture and display
cap = cv2.VideoCapture(0) 
ret, frame = cap.read()
image_size_x = frame.shape[1]
image_size_y = frame.shape[0]


# FIXME: Choose correct com-port and device
if enable_MC:
    servo = maestro.Controller('COM5', device=1)  # Phyz; Check COM port in Windows Device Manager
    #servo = maestro.Controller('COM3', device=2)  # Keith @ home; Check COM port in Windows Device Manager


clock = pygame.time.Clock()

# Set head and arms (in image) to middle
pos_x = image_size_x//2
pos_y = image_size_y//2
head_angle = 0
left_arm = 0         
right_arm = 0


# Set head servos to nominal
if enable_MC:
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
    servo.setAccel(head_x_channel, accel)
    servo.setAccel(head_y_channel, accel)
    servo.setAccel(head_tilt_channel, accel)
    servo.setAccel(arm_left_channel, accel)
    servo.setAccel(arm_right_channel, accel)




#num_people = 3
if num_people > 0:
    random_people_list = choose_people_locations(num_people)   # FIXME: random number of people???
else:
    random_people_list = []
#print(people_list)

looking_at_person = False
person_num = 0

people_list = []
time_to_live = 0


while True:
    clock.tick(30)  # Frame Rate = 30 fps
    #print("*** Frame ***")


    # Read the frame from the webcam
    ret, frame = cap.read()

    # Detect Faces and draw on frame
    if enable_face_detect:
        boxes, probs = mtcnn.detect(frame, landmarks=False)
        draw_face_boxes(frame, boxes, probs) #, landmarks)
    else:
        boxes = None

    # Start with detected faces.  Then add some random people if not enough detected
    print("time to live: ", time_to_live)
    if (boxes is None) and (time_to_live > 0):
        pass
        #people_list = [[0,0]]
        time_to_live -= 1
    elif (boxes is None) and (time_to_live <= 0):
        people_list = []
    else:
        time_to_live = FACE_DET_TTL  # number of frames to ignore if no people detected
        people_list = []
        for box, prob in zip(boxes, probs):
            if prob > 0.8:
                x_pos = ((box[0]+box[2])//2 / image_size_x) * 200 - 100
                y_pos = ((box[1]+box[3])//2 / image_size_y) * 200 - 100
                people_list.append((x_pos,y_pos))
    if len(people_list) < num_people:
        delta = num_people - len(people_list)
        people_list.extend(random_people_list[:delta])
        #print(delta, " people added")

    if len(people_list) == 0:
        people_list = [[0,0]]

    for person in people_list:
        this_x, this_y = get_position(person)
        draw_person_loc(frame, this_x, this_y)

    if not looking_at_person: 
        # Make person 0 most likely
        if np.random.randint(0,100) < 1: # don't switch person, just switch pose
            print("new Pose")
            pass
        #elif np.random.randint(0,100) < 30:   # look at person 0 40% of the time
        #    person_num = 0
        #    print("person = ", person_num)
        else:
            person_num = np.random.randint(0,len(people_list))  # 0, num_people
        #person_duration_count = int(np.random.normal(40,12)+5)  # num of frames to keep looking at this person
        person_duration_count = 7  # num of frames to keep looking at this person
        looking_at_person = True
        pos_x, pos_y = get_position(people_list[person_num])
        #print("pos_x changed")
        head_angle = int(np.random.normal(0, 25))
        if np.random.randint(0,100) < 10: # hands up
            arm_left_axis = 0
            arm_right_axis = 1
            #person_duration_count = 5
            print('hands up!!!!')
        else:
            arm_left_axis = abs((np.random.normal(0.4, 0.3)))
            arm_right_axis = abs((np.random.normal(0.1, 0.3)))
        #print(arm_left_axis, arm_right_axis)
    elif looking_at_person and person_duration_count > 0:
        person_duration_count -= 1
    else:
        looking_at_person = False


    #print("person = ", person_num)
    #print("duration: ", person_duration_count)

    events = pygame.event.get()
    
    if enable_GUI:
        draw_pos(frame, pos_x, pos_y, head_angle, arm_left_axis, arm_right_axis)
        #print(pos_x, pos_y)
        cv2.imshow('image', frame) 
    
    #print("Done")

    if enable_MC:
        move_physical_position(people_list[person_num], head_angle, arm_left_axis, arm_right_axis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
print("EXITING NOW")
exit()
