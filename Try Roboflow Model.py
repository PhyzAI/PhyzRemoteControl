# Try Roboflow model

from inference.models.utils import get_roboflow_model
import cv2

# Roboflow model
#model_name = "face-detection-mik1i"
#model_version = "18"

model_name = "phyzobjectdetection"
model_version = "1"


model = get_roboflow_model(
    model_id="{}/{}".format(model_name, model_version),
    #Replace ROBOFLOW_API_KEY with your Roboflow API Key
    api_key="lxBY3kkYRdVFO7EFWa1M"
)



# Open the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was read successfully, display it
    if ret:
        # Run inference on the frame
        results = model.infer(image=frame,
                        confidence=0.5,
                        iou_threshold=0.5)

        # Plot image with face bounding box (using opencv)
        if results[0].predictions:
            prediction = results[0].predictions[0]
            print(prediction)

            x_center = int(prediction.x)
            y_center = int(prediction.y)
            width = int(prediction.width)
            height = int(prediction.height)

            # Calculate top-left and bottom-right corners from center, width, and height
            x0 = x_center - width // 2
            y0 = y_center - height // 2
            x1 = x_center + width // 2
            y1 = y_center + height // 2

            
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255,255,0), 10)
            cv2.putText(frame, "Face", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)

        # Press 'q' to quit the video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error: Could not read frame.")
        break

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()