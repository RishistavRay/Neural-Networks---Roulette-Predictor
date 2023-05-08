import cv2
import cvzone
from cvzone.ColorModule import ColorFinder

#initialize the video
cap = cv2.VideoCapture('Videos/over.mp4')


#create colorfinder obj

myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 0, 'smin': 0, 'vmin': 117, 'hmax': 179, 'smax': 81, 'vmax': 255}

# List of Frames

posList = []

while True:
    success, img = cap.read()
    #img = cv2.imread("frame.png")
    img = img[50:,:]




    # find color of the ball
    imgColor, mask = myColorFinder.update(img,hsvVals)

    imgContours, contours = cvzone.findContours(img,mask,minArea = 500)

    #Now we will display the position of the ball on a frame by frame basis

    if contours:
        posList.append(contours[0]['center'])
        cx, cy = contours[0]['center']
        print(cx, cy)

        for pos in posList:
            cv2.circle(imgContours,pos,5,(0,0,255),cv2.FILLED) #printing each circle;


    #Display the information
    imgColor = cv2.resize(imgContours, (0, 0), None, 1, 1)
    #cv2.imshow("Image",img)
    cv2.imshow("ImageColor", imgContours)




    cv2.waitKey(5)