# PhyzAI remote control, with sort-of random head positinos and poses
# with an on-screen display of position
#
# Initial Rev, RKD 2024-08

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
import zmaestro as maestro
import cv2 
import numpy as np


# Image Definitions

enable_GUI = True
enable_MC = False   # enable Motor Control

image_size_x = 800
image_size_y = 600



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



def draw_pos(pos_x,pos_y, angle=0, left_arm=0, right_arm=0): 
    """ Draw an image of the current head and arm positions """

    image = np.zeros((image_size_y,image_size_x,3), dtype=np.uint8)

    # Getting the height and width of the image 
    height = image.shape[0] 
    width = image.shape[1] 
    
    # Drawing the lines 
    cv2.line(image, (0, 0), (width, height), (0, 0, 255), 5) 
    cv2.line(image, (width, 0), (0, height), (0, 0, 255), 5) 

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
        
    # Show the image 
    cv2.imshow('image', image) 
  

def choose_people_locations(num_people = 5):
    # return a list of people, where each pair shows percentage of total range available
    people_list = np.random.randint(-100, 100, (num_people,2))
    people_list[0] = [0,0]
    return people_list
     
def get_position(person_loc = [0,0]):
    # Translate relative person location to point on the screen
    x_loc = person_loc[0]
    y_loc = person_loc[1]

    x_scale = int(0.8*image_size_x / 2)
    y_scale = image_size_y // 2 // 2  # don't look too much up or down

    x_pos = int(image_size_x/2 + x_loc*x_scale/100)
    y_pos = int(image_size_y/2 + y_loc*y_scale/100)
    return(x_pos,y_pos)

def move_physical_position(person_loc=[0,0], angle=0, left_arm=0, right_arm=0):
    # Translate relative person location and head/arms to physical position
    x_loc = person_loc[0]
    y_loc = person_loc[1]
    x_scale = int(0.8*(head_x_range[2] - head_x_range[0])/2 )
    y_scale = (head_y_range[2] - head_y_range[0])//2 // 2  # don't look too much up or down
    x_pos = int(head_x_range[1] + x_loc*x_scale/100)
    y_pos = int(head_y_range[1] + y_loc*y_scale/100)

    angle_scale = (head_tilt_range[2] - head_tilt_range[0])//2
    head_pos = int(head_tilt_range[1] + angle*angle_scale/45)

    arm_left_scale = (arm_left_range[2] - arm_left_range[0])//2
    arm_right_scale = (arm_right_range[2] - arm_right_range[0])//2
    arm_left_pos = int(arm_left_range[1] + left_arm*arm_left_scale)
    arm_right_pos = int(arm_right_range[1] + right_arm*arm_right_scale)

    if enable_MC:
        servo.setTarget(head_x_channel, x_pos)
        servo.setTarget(head_y_channel, y_pos)
        servo.setTarget(head_tilt_channel, head_pos)
        servo.setTarget(arm_left_channel, arm_left_pos)
        servo.setTarget(arm_right_channel, arm_right_pos)

    # else:
    #     print(x_pos, head_x_range, y_pos, head_y_range) #head_pos, arm_left_pos, arm_right_pos)

    #return(x_pos, y_pos, head_pos, arm_left_pos, arm_right_pos)
    return


############
### MAIN ###
############

pygame.init()

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

num_people = 11
people_list = choose_people_locations(num_people)   # FIXME: random number of people???
print(people_list)

looking_at_person = False
person_num = 0

try:
    while True:
        clock.tick(30)  # Frame Rate = 30 fps

        if not looking_at_person: 
            # Make person 0 most likely
            if np.random.randint(0,100) < 40: # don't switch person, just switch pose
                print("new Pose")
                pass
            elif np.random.randint(0,100) < 50:   # look at person 0 40% of the time
                person_num = 0
                print("person = ", person_num)
            else:
                person_num = np.random.randint(1,num_people)
                print("person = ", person_num)
            person_duration_count = int(np.random.normal(40,12)+5)  # num of frames to keep looking at this person
            #print("duration: ", person_duration_count)
            looking_at_person = True
            pos_x, pos_y = get_position(people_list[person_num])
            head_angle = int(np.random.normal(0, 10))
            if np.random.randint(0,100) < 3: # hands up
                arm_left_axis = 1
                arm_right_axis = 1
                #person_duration_count = 5
                print('hands up!!!!')
            else:
                arm_left_axis = abs((np.random.normal(0.2, 0.2)))
                arm_right_axis = abs((np.random.normal(0.2, 0.3)))
            #print(arm_left_axis, arm_right_axis)
        elif looking_at_person and person_duration_count > 0:
            person_duration_count -= 1
        else:
            looking_at_person = False
            

        events = pygame.event.get()
        

        if enable_GUI:
            draw_pos(pos_x, pos_y, head_angle, arm_left_axis, arm_right_axis)

        move_physical_position(people_list[person_num], head_angle, arm_left_axis, arm_right_axis)
        
        
except KeyboardInterrupt:
    print("EXITING NOW")
    #j.quit()