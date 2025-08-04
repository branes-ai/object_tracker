import cv2

print("BRANES.AI - Webcam Capture: Exit on 'q' key press")

Width  = 640
Height = 480

writer = cv2.VideoWriter('webcam_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (Width, Height))

cap = cv2.VideoCapture(0)  # Open the webcam
while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break  # Exit if no frame is captured
    cv2.imshow('BRANES.AI', frame)  # Display the frame
    writer and writer.write(frame)  # Write the frame to the video file
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
        break

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows