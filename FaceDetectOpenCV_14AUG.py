
# import libraries
import cv2

# loading the casecades
# download haarcascade from https://github.com/opencv/opencv/tree/master/data/haarcascades
cascade_f=cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # for face
cascade_e=cv2.CascadeClassifier('haarcascade_eye.xml') # for eye

# define a function which will take gray and color image as imput and returns the image with rectangular detectors for face and eye
# as cascade works on gray sclae image we will input transformed gray scale image to detector function. But at the end we will transform the detections to original color image
def detector(gray, color):
    face=cascade_f.detectMultiScale(gray, 1.4, 7) 
    # apply detectMultiScale method from cascase_f to detect faces in the image
    # detecMultiScale will get us the cordinates of face
    # arguments of the method: 1.black and white image i.e. gray here, 2.scaled factor: by how much the size of the image will be changed, 3. min num of neighobours: in order a pic to be accepted at least min neighbour to be also accepted[here 7]
    # 1.4 and 5 are giving better result, but can be tried with other values also
    # face is tuple of 4 elements: cordinates of upper left coorner(x,y), wdith(w) and height(h) of rectange
    for (x, y, w, h) in face: # for loop for each rectangle
        cv2.rectangle(color, (x, y), (x+w, y+h), (255, 0, 0), 2) # draw the rectangle
        # rectangle arguments: 1.image, 2.cordinate of upper left corner, 3. lower right corner of rectangle, 4. color: RGB code for Red (255, 0, 0), 5. thickness of edge of rectangle (2)
        reg_gray= gray[y:y+h, x:x+w] # region of intrest in grayscale image
        reg_color= color[y:y+h, x:x+w] # region of intrest in color image
        eye=cascade_e.detectMultiScale(reg_gray,1.1, 4) # detect eye in gray scale image. Scale and num of min neighbours can be changed. 1.1 and 5 based on experimentation
        for (ex,ey,ew,eh) in eye: 
            # eye for loop inside the face region as we are drawning rectangle for eye in reference to rectangel for face
            # ex, ey, ew and ey are rectanle colrdinate
            cv2.rectangle(reg_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
            # draw rectangle in color region, RGB color Green i.e. 0,255,0
    return color # retrun the image with detector rectangle

# turn on the webcam with VideoCapture. O for inbuilt webcam and 1 for external webcam
video_capture=cv2.VideoCapture(0) 

# start a while loop infinitely untill break
while True:
    _, color = video_capture.read() 
    # use read method to get the the last frame. _ used not to get the 1st frame
    gray=cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) 
    # cvtColor transform the color image to grayscale
    # cvtColor arguments: 1. image, 2. openCV tranform method on color image to get graysclae image
    screen=detector(gray, color) # apply detector function
    cv2.imshow('Video', screen) # display processed image in an animated way on screen
    if cv2.waitKey(1) & 0xFF==ord('q'): # stop webcam if 'q' key is pressed
        break

# turnoff the webcam
video_capture.release()
# destroy all windows where image is displayed
cv2.destroyAllWindows()