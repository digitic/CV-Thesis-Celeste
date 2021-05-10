import numpy
import cv2

#Determine source video
cap = cv2.VideoCapture("CelesteVideos/Level1.mp4")

frame_width = int(cap.get(3) / 2)
frame_height = int(cap.get(4) / 2)

#Output file creation
out = cv2.VideoWriter('CelesteVideos/edgeDetection.mp4',cv2.VideoWriter_fourcc('M','P','4','V'), 30, (frame_width,frame_height), False)

while(cap.isOpened()):
    #ret is a boolean of if a frame was successfully captured
    ret, frame = cap.read()
    #frameTaken = (frameTaken + 1) % FRAMEGAP
    if (ret): #and frameTaken == 0:
        frame = cv2.pyrDown(frame)

        #Edge detection test
        edges = cv2.Canny(frame, 200, 450)
        """
        width = int(edges.shape[1] * scale_percent / 100)
        height = int(edges.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized_edges = cv2.resize(edges, dim, interpolation = cv2.INTER_AREA)
        """
        out.write(edges)
        #cv2.imshow('Canny edge detection', edges)

    #Break if capture ends.
    else:
        break
    #Break if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()