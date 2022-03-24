import cv2
import mouse
hand_width=10
focal=#YOUR cameras focal length
cascadePath = "your fist detecting harrcascade's location"
faceCascade = cv2.CascadeClassifier(cascadePath)
cam = cv2.VideoCapture(0)
cam.set(3, 1535)#my laptop's screen size in pixels
cam.set(4, 863)
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
	distance = (real_face_width * Focal_Length)/face_width_in_frame
	return distance
while True:
        ret, img =cam.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )
        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            mtx=round(x*1)
            mty=round(y*1)
            mx,my=mouse.get_position()
            dw=w
            mouse.move(mtx-mx, mty-my, absolute=False, duration=0.2)
            nowdis=Distance_finder(focal,hand_width,dw)
            if(nowdis<40):
               mouse.click()
        cv2.imshow('camera',img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:#x=640,y=480
            break
