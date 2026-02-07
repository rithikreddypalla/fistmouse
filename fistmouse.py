
"""
FistMouse: Control your mouse with your fist using computer vision.
---------------------------------------------------------------
This script uses OpenCV and a Haar cascade to detect a fist and moves the mouse pointer accordingly.
If the fist is close to the camera, it triggers a mouse click.

Configuration:
- Set the correct path to your Haar cascade XML for fist detection.
- Set your camera's focal length and screen size.
- Requires the 'mouse' and 'opencv-python' packages.
"""

import cv2
import mouse

# --- CONFIGURATION ---
HAND_WIDTH = 10  # Real width of your fist in cm (approximate)
FOCAL_LENGTH = 500  # Set your camera's focal length in pixels (calibrate for accuracy)
CASCADE_PATH = "fist.xml"  # Path to your Haar cascade for fist detection
SCREEN_WIDTH = 1535  # Set to your screen width in pixels
SCREEN_HEIGHT = 863  # Set to your screen height in pixels

# --- SETUP ---
fist_cascade = cv2.CascadeClassifier(CASCADE_PATH)
cam = cv2.VideoCapture(0)
cam.set(3, SCREEN_WIDTH)
cam.set(4, SCREEN_HEIGHT)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

def distance_finder(focal_length, real_width, width_in_frame):
    """Estimate distance from camera using similar triangles."""
    if width_in_frame == 0:
        return float('inf')
    return (real_width * focal_length) / width_in_frame

try:
    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fists = fist_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )
        for (x, y, w, h) in fists:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Move mouse pointer to the detected fist position
            target_x = round(x)
            target_y = round(y)
            current_x, current_y = mouse.get_position()
            mouse.move(target_x - current_x, target_y - current_y, absolute=False, duration=0.2)
            # Estimate distance and click if close
            distance = distance_finder(FOCAL_LENGTH, HAND_WIDTH, w)
            if distance < 40:
                mouse.click()
        cv2.imshow('FistMouse Camera', img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:  # ESC to exit
            break
except Exception as e:
    print(f"Error: {e}")
finally:
    cam.release()
    cv2.destroyAllWindows()
