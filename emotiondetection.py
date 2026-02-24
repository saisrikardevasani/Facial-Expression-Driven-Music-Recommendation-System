import cv2

# Initialize webcam (0 is the default camera)
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Process the frame here

        # Show the frame
        cv2.imshow("Webcam", frame)

        # Break loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
webcam.release()
cv2.destroyAllWindows()
