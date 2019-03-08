
from tensorflow_detection import DetectionObj
from threading import Thread

import os
import cv2

def resize(image, new_width=None, new_heigh=None):
    """
    Resize an image based on a new width or new height
    keeping the original ratio
    """
    height, width, depth = image.shape
    if new_width:
        new_heigh = int((new_width /float(width))*height)
    elif new_heigh:
        new_width = int((new_heigh/float(height))*width)
    else:
        return image
    return cv2.resize(image, (new_width, new_heigh), interpolation=cv2.INTER_AREA)

class WebcamStream:
    def __init__(self):
        # Initialize Webcam
        self.stream = cv2.VideoCapture(-1)
        # Starting TensorFlow API with SSD Mobilenet
        self.detection = DetectionObj(model='ssd_inception_v2_coco_11_06_2017')
        # Starting capturing video so the Webcam will tune itself
        _, self.frame = self.stream.read()
        # Set the stop flag to False
        self.stop = False
        #
        Thread(target=self.refresh, args=()).start()

        # Record the video on Webcam
        self._fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.saving_path = os.path.join(self.detection.CURRENT_PATH, 'webcam_videos', 'webcam' + self.detection.get_time() + '.avi')
        self.out = cv2.VideoWriter(self.saving_path, self._fourcc, 10, (640,480))


    def refresh(self):
        # Looping until an explicit stop is sent
        # from outside the function
        while True:
            if self.stop:
                return
            _, self.frame = self.stream.read()

    def get(self):
        # returning the annotated image
        return self.detection.annotate_photogram(self.frame)

    def halt(self):
        # Setting the halt flag
        self.stop = True

if __name__ == "__main__":
    stream = WebcamStream()
    while True:
        # Grabbing the frame from the threaded video stream
        # and resize it to have a maximum width of 640 pixels
        frame = resize(stream.get(), new_width=640)
        #print(frame.shape)
        stream.out.write(frame)
        cv2.imshow("Webcam", frame)
        #If the space bar is hit, the program will stop
        if cv2.waitKey(1) & 0xFF == ord(" "):
            # First stopping the streaming thread
            stream.halt()
            # Then halting the while loop
            break
    stream.out.release()
    cv2.destroyAllWindows()
