# NOTA: el pinch cushion me parece que no está muy allá, ya lo revisaré luego

import cv2
import numpy as np
import time

def nothing(x):
    pass

# Global variable to trigger capture
capture_flag = False

def mouse_callback(event, x, y, flags, param):
    global capture_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is within the button area (e.g., top-left 100x50)
        if 10 <= x <= 110 and 10 <= y <= 60:
            capture_flag = True

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a window for the camera and trackbars
cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', mouse_callback)

# Create trackbars for distortion parameters K1 and K2
# Range: 0 to 200, offset 100 to allow negative values (-100 to 100)
cv2.createTrackbar('K1 (x10^-7)', 'Camera', 100, 200, nothing) 
cv2.createTrackbar('K2 (x10^-12)', 'Camera', 100, 200, nothing)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# Pre-calculate coordinates for the map
xcen = frame_width / 2
ycen = frame_height / 2
x_grid, y_grid = np.meshgrid(np.arange(frame_width), np.arange(frame_height))
xd_rel = x_grid - xcen
yd_rel = y_grid - ycen
r2 = xd_rel**2 + yd_rel**2
r4 = r2**2

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Get trackbar positions and scale them (centered at 100)
    k1 = (cv2.getTrackbarPos('K1 (x10^-7)', 'Camera') - 100) * 1e-7
    k2 = (cv2.getTrackbarPos('K2 (x10^-12)', 'Camera') - 100) * 1e-12

    # Calculate the undistorted mapping
    map_x = x_grid + xd_rel * (k1 * r2 + k2 * r4)
    map_y = y_grid + yd_rel * (k1 * r2 + k2 * r4)

    # Apply remap
    distorted_frame = cv2.remap(frame, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)

    # Drawing the "Capture" button overlay
    # Rectangle (x1, y1), (x2, y2), color, thickness (-1 for fill)
    cv2.rectangle(distorted_frame, (10, 10), (110, 60), (0, 0, 0), -1)
    cv2.putText(distorted_frame, "CAPTURE", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if capture_flag:
        filename = f"capture_{int(time.time())}.png"
        cv2.imwrite(filename, distorted_frame)
        print(f"Image saved as {filename}")
        # Brief visual feedback (invert button color)
        cv2.rectangle(distorted_frame, (10, 10), (110, 60), (0, 255, 0), -1)
        cv2.putText(distorted_frame, "SAVED!", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        capture_flag = False

    # Write the frame to the output file
    out.write(distorted_frame)

    # Display the distorted frame
    cv2.imshow('Camera', distorted_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()