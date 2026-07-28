# conda activate cv_env

import cv2

def main():
    # Initialize the webcam (0 is usually the default built-in camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create the Spectral Residual saliency detector
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

    print("Press 'q' to exit the stream.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Compute the saliency map
        # success is a boolean, saliencyMap is a float array with values in [0, 1]
        success, saliencyMap = saliency.computeSaliency(frame)

        if success:
            # Convert float map [0.0, 1.0] to an 8-bit image [0, 255] for display
            saliencyMap = (saliencyMap * 255).astype("uint8")

            # Apply a color map for better visual representation of salient areas
            saliency_heatmap = cv2.applyColorMap(saliencyMap, cv2.COLORMAP_JET)

            # Display the original feed and the saliency map side-by-side
            combined_view = cv2.hconcat([frame, saliency_heatmap])
            cv2.imshow("Webcam Stream (Left) vs. Saliency Map (Right)", combined_view)

        # Press 'q' to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



    