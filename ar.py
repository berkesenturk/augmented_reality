import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Global variables for the OpenGL window
window = None
marker_found = False
rotation_angle = 0

# Define the AR marker ID to track
AR_MARKER_ID = 0

# ARUCO marker setup
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

# Camera parameters (dummy values, replace with calibration if available)
camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))

# OpenGL initialization
def init_opengl():
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glEnable(GL_DEPTH_TEST)

# Function to draw a 3D rectangle with example text
def draw_3d_rectangle():
    global rotation_angle
    glPushMatrix()
    glTranslatef(0.0, 0.0, -5.0)
    glRotatef(rotation_angle, 0.0, 1.0, 0.0)
    
    # Draw the rectangle
    glColor3f(0.5, 0.8, 0.5)
    glBegin(GL_QUADS)
    glVertex3f(-1.0, -1.0, 0.0)
    glVertex3f(1.0, -1.0, 0.0)
    glVertex3f(1.0, 1.0, 0.0)
    glVertex3f(-1.0, 1.0, 0.0)
    glEnd()

    # Add example text
    glColor3f(1.0, 1.0, 1.0)
    glRasterPos3f(-0.5, 0.0, 0.01)
    for c in "Example Text":
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(c))

    glPopMatrix()

# OpenGL display function
def display():
    global marker_found, rotation_angle

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    if marker_found:
        draw_3d_rectangle()
        rotation_angle += 1  # Rotate the rectangle

    glutSwapBuffers()

# Update function for GLUT
def update(value):
    glutPostRedisplay()
    glutTimerFunc(16, update, 0)

# Function to process the video feed
def process_video():
    global marker_found

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None and AR_MARKER_ID in ids:
            marker_found = True
        else:
            marker_found = False

        # Show the video feed
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow("Video Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    global window

    # Initialize OpenGL
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    window = glutCreateWindow(b"AR Marker Detection")
    init_opengl()

    # Set GLUT callbacks
    glutDisplayFunc(display)
    glutTimerFunc(16, update, 0)

    # Run video processing in a separate thread
    import threading
    video_thread = threading.Thread(target=process_video)
    video_thread.daemon = True
    video_thread.start()

    # Start the GLUT main loop
    glutMainLoop()

if __name__ == "__main__":
    main()
