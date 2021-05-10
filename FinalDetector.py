import numpy
import cv2

FRAMEGAP = 8

#Determine source video
cap = cv2.VideoCapture("CelesteVideos/Level1Trim.mp4")

frame_width = int(cap.get(3) / 2)
frame_height = int(cap.get(4) / 2)

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#Reference images
spikes = cv2.imread("edge-images/celeste_new_spikes.jpg")
spikes = cv2.resize(spikes, (30, 15))
h_spikes, w_spikes = spikes.shape[0:2]

cv2.imshow("blah", spikes)

right_spikes = cv2.rotate(spikes, cv2.cv2.ROTATE_90_CLOCKWISE)

down_spikes = cv2.imread("edge-images/celeste_down_spikes.jpg")
down_spikes = cv2.resize(down_spikes, (30, 14))
#h_spikes, w_spikes = spikes.shape[0:2]

left_spikes = cv2.rotate(down_spikes, cv2.cv2.ROTATE_90_CLOCKWISE)

powerup = cv2.imread("edge-images/celeste_new_powerup.jpg")
powerup = cv2.resize(powerup, (30, 30))
h_powerup, w_powerup = powerup.shape[0:2]

spring = cv2.imread("edge-images/celeste_new_spring.jpg")
spring = cv2.resize(spring, (40, 10))
h_spring, w_spring = spring.shape[0:2]

berry = cv2.imread("edge-images/celeste_new_strawberry.jpg")
berry = cv2.resize(berry, (30, 40))
h_berry, w_berry = berry.shape[0:2]
"""
playerIdle = cv2.imread("edge-images/celeste_player_idle.jpg")
playerIdle = cv2.resize(playerIdle, (30, 40))
h_playerIdle, w_playerIdle = playerIdle.shape[0:2]

playerJump = cv2.imread("edge-images/celeste_player_jump.jpg")
playerJump = cv2.resize(playerJump, (30, 40))
h_playerJump, w_playerJump = playerJump.shape[0:2]
"""
stoplight = cv2.imread("edge-images/celeste_stoplight.jpg")
stoplight = cv2.resize(stoplight, (20, 40))
h_stoplight, w_stoplight = stoplight.shape[0:2]

greenlight = cv2.imread("edge-images/celeste_greenlight.jpg")
greenlight = cv2.resize(greenlight, (20, 40))
h_greenlight, w_greenlight = greenlight.shape[0:2]

yellowlight = cv2.imread("edge-images/celeste_yellowlight.jpg")
yellowlight = cv2.resize(greenlight, (20, 40))
h_yellowlight, w_yellowlight = yellowlight.shape[0:2]

plat = cv2.imread("edge-images/celeste_1d_platform.jpg")
plat = cv2.resize(plat, (30, 10))
h_plat, w_plat = plat.shape[0:2]

gear = cv2.imread("edge-images/celeste_gear.jpg")
gear = cv2.resize(gear, (30, 30))
h_gear, w_gear = gear.shape[0:2]
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#We'll check only every FRAMEGAP frame
frameTaken = FRAMEGAP - 1

#Output file creation
#out = cv2.VideoWriter('CelesteVideos/objDetection.mp4',cv2.VideoWriter_fourcc('M','P','4','V'), 30, (frame_width,frame_height))

#While source video is open, take in each frame as a screenshot.
while(cap.isOpened()):
    #ret is a boolean of if a frame was successfully captured
    ret, frame = cap.read()
    frameTaken = (frameTaken + 1) % FRAMEGAP
    if (ret) and frameTaken == 0:
        frame = cv2.pyrDown(frame)
        #Create a grayscale frame out of this for use
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #Determine points where the reference image for up-pointing spikes closely resembles a location in the video frame.
        res = cv2.matchTemplate(frame, spikes, cv2.TM_CCOEFF_NORMED)
        threshold = 0.68
        loc = numpy.where( res >= threshold)

        #Draw blue rectangles around matching points, display color version
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w_spikes, pt[1] + h_spikes), (255,0,0), 2)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #Determine points where the reference image for right-pointing spikes closely resembles a location in the video frame.
        res = cv2.matchTemplate(frame, right_spikes, cv2.TM_CCOEFF_NORMED)
        threshold = 0.68
        loc = numpy.where( res >= threshold)

        #Draw blue rectangles around matching points, display color version
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + h_spikes, pt[1] + w_spikes), (255,0,0), 2)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #Determine points where the reference image for down-pointing spikes closely resembles a location in the video frame.
        res = cv2.matchTemplate(frame, down_spikes, cv2.TM_CCOEFF_NORMED)
        threshold = 0.68
        loc = numpy.where( res >= threshold)

        #Draw blue rectangles around matching points, display color version
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w_spikes, pt[1] + h_spikes), (255,0,0), 2)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #Determine points where the reference image for left-pointing spikes closely resembles a location in the video frame.
        res = cv2.matchTemplate(frame, left_spikes, cv2.TM_CCOEFF_NORMED)
        threshold = 0.68
        loc = numpy.where( res >= threshold)

        #Draw blue rectangles around matching points, display color version
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + h_spikes, pt[1] + w_spikes), (255,0,0), 2)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        #Determine points where the reference image for powerups closely resembles a location in the video frame.
        res = cv2.matchTemplate(frame, powerup, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = numpy.where( res >= threshold)

        #Draw green rectangles around matching points, display color version
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w_powerup, pt[1] + h_powerup), (0,255,0), 2)
        
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #Determine points where the reference image for springboards closely resembles a location in the video frame.
        res = cv2.matchTemplate(frame, spring, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = numpy.where( res >= threshold)

        #Draw orange rectangles around matching points, display color version
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w_spring, pt[1] + h_spring), (0,165,255), 2)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #Determine points where the reference image for strawberries closely resembles a location in the video frame.
        res = cv2.matchTemplate(frame, berry, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = numpy.where( res >= threshold)

        #Draw red rectangles around matching points, display color version
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w_berry, pt[1] + h_berry), (0,0,255), 2)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #Determine points where the reference image for stoplights closely resembles a location in the video frame.
        res = cv2.matchTemplate(frame, stoplight, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = numpy.where( res >= threshold)

        #Draw yellow rectangles around matching points, display color version
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w_stoplight, pt[1] + h_stoplight), (0,255,255), 2)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #Determine points where the reference image for green lights closely resembles a location in the video frame.
        res = cv2.matchTemplate(frame, greenlight, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = numpy.where( res >= threshold)

        #Draw yellow rectangles around matching points, display color version
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w_greenlight, pt[1] + h_greenlight), (0,255,255), 2)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #Determine points where the reference image for yellow lights closely resembles a location in the video frame.
        res = cv2.matchTemplate(frame, yellowlight, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = numpy.where( res >= threshold)

        #Draw yellow rectangles around matching points, display color version
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w_yellowlight, pt[1] + h_yellowlight), (0,255,255), 2)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #Determine points where the reference image for one-sided platforms closely resembles a location in the video frame.
        res = cv2.matchTemplate(frame, plat, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = numpy.where( res >= threshold)

        #Draw brown rectangles around matching points, display color version
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w_plat, pt[1] + h_plat), (42,42,140), 2)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #Determine points where the reference image for gears closely resembles a location in the video frame.
        res = cv2.matchTemplate(frame, gear, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = numpy.where( res >= threshold)

        #Draw dark yellow rectangles around matching points, display color version
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w_gear, pt[1] + h_gear), (55,255,255), 2)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
        """
        #Determine points where the reference image for the idle player closely resembles a location in the video frame.
        res = cv2.matchTemplate(frame, playerIdle, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = numpy.where( res >= threshold)

        #Draw pink rectangles around matching points, display color version
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w_playerIdle, pt[1] + h_playerIdle), (200,200,255), 2)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #Determine points where the reference image for the jumping player closely resembles a location in the video frame.
        res = cv2.matchTemplate(frame, playerJump, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = numpy.where( res >= threshold)

        #Draw pink rectangles around matching points, display color version
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w_playerJump, pt[1] + h_playerJump), (200,200,255), 2)
        """
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #Display frame
        cv2.imshow('Input Video', frame)
        #out.write(frame)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Edge detection
        """
        edges = cv2.Canny(frame, 100, 200)

        cv2.imshow("Edge Detection", edges)
        """
    #Break if capture ends.
    elif not ret:
        break
    #Break if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#out.release()
cv2.destroyAllWindows()