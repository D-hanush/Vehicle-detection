import cv2
import numpy as np

# Initialize video capture from the given video file
cap = cv2.VideoCapture('car.mp4')

# Minimum width and height of the detected rectangle to be considered as a valid detection
min_width_react = 80  # Minimum width of the rectangle
min_height_react = 80  # Minimum height of the rectangle

# Position of the line used for vehicle counting
count_line_position = 550

# Initialize background subtractor using the MOG method
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

# Function to get the center of a rectangle
def center_handle(x, y, w, h):
    """
    Calculate the center of a rectangle.

    Args:
        x (int): x-coordinate of the top-left corner
        y (int): y-coordinate of the top-left corner
        w (int): width of the rectangle
        h (int): height of the rectangle

    Returns:
        tuple: (cx, cy) coordinates of the center
    """
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# List to store the center points of detected rectangles
detect = []
# Allowable error in pixel for line crossing detection
offset = 6
# Counter for the number of vehicles
counter = 0

while True:
    # Read a new frame from the video
    ret, frame1 = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to the grayscale frame
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    # Apply the background subtractor on the blurred frame
    img_sub = algo.apply(blur)
    # Dilate the image to fill in holes and gaps
    dilat = cv2.dilate(img_sub, np.ones((5, 5)), iterations=1)
    # Use morphological closing to further reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    # Find contours on the processed image
    counterShape, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the counting line on the original frame
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

    for (i, c) in enumerate(counterShape):
        # Get bounding box coordinates for each contour
        (x, y, w, h) = cv2.boundingRect(c)
        # Validate if the bounding box meets the size criteria
        validate_counter = (w >= min_width_react) and (h >= min_height_react)
        if not validate_counter:
            continue
        
        # Draw the bounding rectangle on the frame
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Get the center of the bounding rectangle
        center = center_handle(x, y, w, h)
        detect.append(center)
        # Draw a circle at the center point
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)
        
    for (cx, cy) in detect:
        # Check if the center point is within the line offset range
        if cy < (count_line_position + offset) and cy > (count_line_position - offset):
            counter += 1
            detect.remove((cx, cy))
            print("Vehicle Counter:" + str(counter))
            
    # Display the vehicle count on the frame
    cv2.putText(frame1, "VEHICLE COUNTER : " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # Show the processed video frame
    cv2.imshow('Video Original', frame1)

    # Break the loop if 'ESC' key is pressed
    if cv2.waitKey(30) == 27:  # Press 'ESC' to exit
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
