import cv2
import numpy as np

from fdlite import FaceDetection, FaceDetectionModel
from fdlite.render import Colors, detections_to_render_data, render_to_image 
from PIL import Image


"""
FaceDetectionModel.FRONT_CAMERA - a smaller model optimised for selfies and close-up portraits; this is the default model used
FaceDetectionModel.BACK_CAMERA - a larger model suitable for group images and wider shots with smaller faces
FaceDetectionModel.SHORT - a model best suited for short range images, i.e. faces are within 2 metres from the camera
FaceDetectionModel.FULL - a model best suited for mid range images, i.e. faces are within 5 metres from the camera
FaceDetectionModel.FULL_SPARSE - a model best suited for mid range images, i.e. faces are within 5 metres from the camera
"""

detect_faces = FaceDetection(model_type=FaceDetectionModel.FULL)

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    imH, imW, _ = frame.shape
    
    faces = detect_faces(frame)
    if not len(faces):
        print('no faces detected :(')
    else:
        render_data = detections_to_render_data(faces, bounds_color=Colors.GREEN)
        #render_to_image(render_data, frame).show()
        #print(render_data)
        for face in faces:
            #xmin, ymin, xmax, ymax = face.bbox
            xmin = int(face.bbox.xmin*imW)
            xmax = int(face.bbox.xmax*imW)
            ymin = int(face.bbox.ymin*imH)
            ymax = int(face.bbox.ymax*imH)
            cv2.rectangle(frame,
                                (xmin, ymin),
                                (xmax, ymax),
                                (0, 0, 255),
                                thickness=2)
            prob = face.score
            cv2.putText(frame, str(round(prob,2)),
                         (xmax, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)




    # Show the frame
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

