import numpy as np
import cv2
from mss import mss
from PIL import Image

import time

# My script now accurately tells where the moving images are, I can go from here on out and match templates to the different GUI elements, and reading the map. 
# Once I have done this, I can quantify that data into a sentiment compas and use that to trigger tracks.
  
# mathematical functions in support of others defined here ---------------------------------------------------------------

def mostFrequent(List): 
    mostOccured = max(set(List), key = List.count) 
    accuracy = List.count(mostOccured) / len(List)
    return [mostOccured, accuracy * 100];

# computer vision functions defined below --------------------------------------------------------------------------------

def findVideoWindow(avg):

    # the function below attempts to find the video window that is being used as accurately as possible.
    # We can still optimize the speed at which it is found by limiting the length of videoCoordinates stored, 
    # by not only appending, but also popping.

    # making a copy of the weighted average, to draw shapes on.
    avgDrawn = avg.copy()

    #convert the average into a usable filtype for thresholding
    frameGray = avgDrawn.astype(np.uint8)
    frameGray = cv2.cvtColor(frameGray, cv2.COLOR_BGR2GRAY)


    # you need this for the erosion and dilation
    kernel = np.ones((5,5), np.uint8) 

    # The first parameter is the original image, kernel is the matrix with which image is convolved and third parameter is the number  
    # of iterations, which will determine how much you want to erode/dilate a given image.  
    frameGray = cv2.erode(frameGray, kernel, iterations=5) 
    frameGray = cv2.dilate(frameGray, kernel, iterations=5) 

    
    #finding contours
    ret, thresh = cv2.threshold(frameGray, 0, 255, 0 )
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key = cv2.contourArea)
    
        x,y,w,h = cv2.boundingRect(c)

        # the next steps are looking to optimize the rectangle that has been drawn. The way they do that is by storing all the 
        # found coordinates in the last step, every n frames. And subsequently counting the most occuring element in it. 
        # By dividing this by the total number of found coördinates it gives you an accuracy %. Once a certain leverl of accuracy is µ
        # found we should have the best approximation of our video screen.
        # storing all the coördinates in order to optimize where the video screen is.
        videoCoordinates['x'].append(x)
        videoCoordinates['y'].append(y)
        videoCoordinates['w'].append(w)
        videoCoordinates['h'].append(h)

        optimalCoordinates['x'] = mostFrequent(videoCoordinates['x'])
        optimalCoordinates['y'] = mostFrequent(videoCoordinates['y'])
        optimalCoordinates['w'] = mostFrequent(videoCoordinates['w'])
        optimalCoordinates['h'] = mostFrequent(videoCoordinates['h'])

        accuracyCoordinates = (optimalCoordinates['x'][1] + optimalCoordinates['y'][1] + optimalCoordinates['w'][1] + optimalCoordinates['h'][1]) / 4

        # draw the biggest contour (c) in green
        cv2.rectangle(avgDrawn,(x,y),(x+w,y+h),(0,255,0),2)

        #cv2.drawContours(avgDrawn, contours, -1, (0,255,0), 10)

    return avgDrawn, accuracyCoordinates

# executive function called here, these are basically the functions that sequence and call the opencv functions.

def callVideoFinder(frame, originalBox, avg):

    # counting all the frames that pass allows me to schedule certain functions in sequence.
    frameCount = 0

    # the next boolean represents whether we found the video window with reasonable accuracy
    windowFound = False

    while True:

        #the next line grabs the frames from the bounding box area we defined earlier
        frame2 = frame
        frame1 = sct.grab(originalBox)

        # computes the difference between two frames, in order to get an apporximation of the moving area, in this case hopefully the video screen.
        diff = cv2.absdiff(np.float32(frame1), np.float32(frame2))

        # computes a running average of the moving parts
        cv2.accumulateWeighted(diff, avg, 0.1)

        # below I can schedule the callback of functions based on the amount of frameCounts.
    
        if frameCount % 4 == 0 and windowFound == False: 
            videoBox, accuracy = findVideoWindow(avg)
            print(accuracy)
            if accuracy > 70:
                windowFound = True

                boundingBoxNew = {
                    'top': optimalCoordinates['y'][0] + 100, 
                    'left': optimalCoordinates['x'][0] + 960, 
                    'width': optimalCoordinates['w'][0], 
                    'height': optimalCoordinates['h'][0]
                    }

                cv2.destroyAllWindows()
                return boundingBoxNew

        else:
            videoBox = avg

        cv2.imshow('screen', np.array(frame1))  

        frameCount += 1

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break


time.sleep(1)
# the resolution of my system is 1920 x 1080.
# But because I am working on the same screen I'm goin to catpure a smaller box
# the 100 you're dropping from the top is in order to accomodate the menu bar.
boundingBox = {
    'top': 100, 
    'left': 960, 
    'width': 960, 
    'height': 1080
    }

sct = mss()

# setting first variables before use inside the loop
frame1 = sct.grab(boundingBox)
avg = np.float32(frame1)

# average coördinates
videoCoordinates = {
    'x' : list(range(1,10)),
    'y' : list(range(1,10)),
    'w' : list(range(1,10)),
    'h' : list(range(1,10))
}

# optimal coordinates (coordinates per axis, accuracy level)
optimalCoordinates = {
    'x' : [],
    'y' : [],
    'w' : [],
    'h' : []
}

boundingBox = callVideoFinder(frame1, boundingBox, avg)

while True:

    frame1 = sct.grab(boundingBox)
    frame2 = frame1

    avg = np.float32(frame1)

    cv2.imshow('screen', np.array(frame1))  

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break

