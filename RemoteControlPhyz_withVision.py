# PhyzAI remote control, with face detection
# Alternate version with head-mounted camera
#
# Initial Rev, RKD 2024-08
#
# TODO:
# * add calibration offset to head  -- Check this works!
# * Only look for heads when not moving??
# * Debug: Keep track of head location in Phyz_control_space when camera is head-mounted.
#       This isn't working properly.  The motion of the head messes up face detection.
# * Add some random head moves, and looking around if no face is detected
# * Tweak calibration of Phyz head versus image center
# * add second camera to Phyz's head?  Easier to track exact location of people (closed loop)
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


import pygame
import cv2 
import numpy as np


# Enable different basic operations

enable_GUI = True
enable_MC = True  # enable Motor Control
enable_face_detect = True
enable_head_camera = False  #FIXME: Doesn't work properly

if enable_face_detect:
    from facenet_pytorch import MTCNN


# Calibration to get the head to face you exactly (hopefully)
HEAD_OFFSET_X = 16
HEAD_OFFSET_Y = 0

# FIXME: Choose correct com-port and device
if enable_MC:
    import zmaestro as maestro
    servo = maestro.Controller('COM5', device=1)  # Phyz; Check COM port in Windows Device Manager
    #servo = maestro.Controller('COM3', device=2)  # Keith @ home; Check COM port in Windows Device Manager


num_people = 3   # *Maximum* number of "people" to include in the scene
FACE_DET_TTL = 25  # Hold-time for face detction interruptions (in ticks)

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


def draw_person_loc(image, pos_x, pos_y, color = (0, 100, 0)):
    """ Draw an oval where each (random) face is located """
    axesLength = (20, 40) 
    startAngle = 0
    endAngle = 360
    thickness = 3
    angle = 0
    cv2.ellipse(image, (pos_x, pos_y), axesLength, 
           angle, startAngle, endAngle, color, thickness)
    return image



def draw_pos(image, pos_x,pos_y, angle=0, left_arm=0, right_arm=0): 
    """ Draw an image of the current head and arm positions """

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
        
    return image
  

def choose_people_locations(num_people = 5):
    """return a list of people, where each pair shows percentage of total range available"""
    people_list = np.random.randint(-90, 90, (num_people,2))
    return people_list
     

def get_position(person_loc = [0,0], move_scale = 1.0):
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


def move_physical_position(person_loc=[0,0], angle=0, left_arm=0, right_arm=0, enable_MC = False,
                            move_scale = 1.0, move_relative = False):
    """
    Translate Ideal person location and head/arms to physical position.
    * Ideal is based on a -100 to +100 in x and y dimensions.
    """

    # Get actual camera (head) position
    #print("pos: ", servo.getPosition(head_x_channel), servo.getPosition(head_y_channel))
    if enable_MC:
        head_x = servo.getPosition(head_x_channel)  #FIXME: Convert this to Ideal-plane 
        head_x = (head_x - (head_x_range[0])) / (head_x_range[2]-head_x_range[0])
        head_x = head_x * 200 - 100
        head_y = servo.getPosition(head_y_channel)
        head_y = (head_y - (head_y_range[0])) / (head_y_range[2]-head_y_range[0])
        head_y = head_y * 200 - 100
        #print("head pos (ideal): ", head_x, head_y)
    else:
        head_x = 0
        head_y = 0

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

    #print("x_pos, y_pos: ", x_pos, y_pos)
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

print("*** Starting ***")

pygame.init()

# Create detector    
if enable_face_detect:
    mtcnn = MTCNN()

# Video Capture and display (only 1st 2 backends work on Win11?)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # CAP_MSMF, CAP_DSHOW, _FFMPEG, _GSTREAMER
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print('opened video')
ret, frame = cap.read()
image_size_x = frame.shape[1]
image_size_y = frame.shape[0]

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
    servo.setAccel(head_x_channel, 5*accel)
    servo.setAccel(head_y_channel, 2*accel)
    servo.setAccel(head_tilt_channel, accel)
    servo.setAccel(arm_left_channel, accel)
    servo.setAccel(arm_right_channel, accel)


person_num = 0
people_list = []
time_to_live = 0   # Time that detected faces stay in case of nothing new detected

head_duration_count = 0
body_duration_count = 0

person_offset_x = 0
person_offset_y = 0

while True:
    clock.tick(30)  # Frame Rate = 30 fps

    if enable_MC:
        head_is_moving = (servo.isMoving(head_x_channel) | servo.isMoving(head_y_channel) |
              servo.isMoving(head_tilt_channel) )
    else:
        head_is_moving = False


    # Read the frame from the webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Detect Faces and draw on frame
    if enable_face_detect: # and not head_is_moving:
        boxes, probs = mtcnn.detect(frame, landmarks=False)
        draw_face_boxes(frame, boxes, probs) #, landmarks)
    else:
        boxes = None

    if (boxes is None) and (time_to_live > 0):
        #pass
        #people_list = [[0,0]]
        time_to_live -= 1
    elif (boxes is None) and (time_to_live <= 0):
        people_list = []
        person_num = 0
    else:
        time_to_live = FACE_DET_TTL  # number of frames to ignore if no people detected
        people_list = []
        person_num = 0
        for box, prob in zip(boxes, probs): 
            if prob > 0.90:
                x_box_mid = int((box[0]+box[2])/2)
                y_box_mid = int((box[1]+box[3])/2)
                #frame = cv2.circle(frame, (x_box_mid,y_box_mid), radius=3, color=(0, 0, 255), thickness=3)
                x_pos = (x_box_mid/image_size_x) * 200 - 100
                y_pos = (y_box_mid/image_size_y) * 200 - 100
                people_list.append((x_pos,y_pos))


    if len(people_list) > num_people:
        people_list = people_list[:num_people]
    elif len(people_list) == 0:
        #people_list = [[0,0]]
        this_x = int(np.random.normal(0, 20)) # np.random.randint(-30,30)
        this_y = int(np.random.normal(0, 10)) #np.random.randint(-25,25)
        people_list = [[this_x, this_y]]
        person_num = 0
        time_to_live = FACE_DET_TTL
        #print("random person: ", this_x, this_y)

    for person in people_list:
        this_x, this_y = get_position(person)
        draw_person_loc(frame, this_x, this_y)

    # Look at one person, or switch
    if head_duration_count <= 0:
        # Make person 0 most likely
        if np.random.randint(0,100) < 20:   # Look at the main person
            person_num = 0
            person_offset_x = 0
            person_offset_y = 0
            head_duration_count = int(np.random.normal(10,20)+5)  # num of frames to keep looking at this person
        elif np.random.randint(0,100) < 20:  # Look away a little
            person_offset_x = np.random.randint(-50,50)
            person_offset_y = np.random.randint(-15,15)
            head_duration_count = int(np.random.normal(0,7)+2)  # num of frames to keep looking at this person
            print("Look away: ", person_offset_x, person_offset_y)
        else:   # Switch person
            #print ("Switching person")
            person_num = np.random.randint(0,len(people_list))  # 0, num_people
            person_offset_x = 0
            person_offset_y = 0
            head_duration_count = int(np.random.normal(10,20)+5)  # num of frames to keep looking at this person
    else:
        head_duration_count -= 1

    #print("person pos: ", people_list[person_num])
    # Move Head and arms
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
        


    if enable_head_camera:
            move_scale = 1.0 # 0.7
    else:
            move_scale = 1.0
        
    # FIXME: scale so that new Phyz position is only part of the way to desired.  Prevents overshoot
    # and (hopefully) precessing around the desired spot.
    try:
        this_x, this_y = people_list[person_num]  # FIXME: how does this get out of bounds?????
    except:
        this_x, this_y = (0,0)
        person_num = 0
    this_x *= move_scale
    this_y *= move_scale
    pos_x, pos_y = get_position([this_x, this_y])
    if enable_GUI:
        draw_pos(frame, pos_x, pos_y, head_angle, arm_left_axis, arm_right_axis)
        #print(pos_x, pos_y)
        cv2.imshow('image', frame) 

    # If camera is on the head, only move "move_scale" of the way to the new destination
    # Hopefully, this will prevent overshoot and precessing around the correct location 
    person_x, person_y = people_list[person_num]
    person_x += person_offset_x
    person_y += person_offset_y
    #print(person_offset_x, person_offset_y)
    move_physical_position((person_x, person_y), head_angle, arm_left_axis, arm_right_axis, enable_MC, move_scale, enable_head_camera)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    
    # print("Moving: ", head_is_moving )

    events = pygame.event.get()
    

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
print("EXITING NOW")
exit()
