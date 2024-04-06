import os
import cv2 as cv
import numpy as np

class FrameOperations():

    def __init__(self):
        self.CWD = os.getcwd()
        self.RES_F = os.path.join(self.CWD,'resources')
        self.FILTER_F = os.path.join(self.RES_F,'FILTERS')
        self.SPEED_FILTER = cv.imread(os.path.join(self.FILTER_F,"SPEED.png"))
        self.CONT_FILTER = cv.imread(os.path.join(self.FILTER_F,"CONTINUE.png"))


    def found_frame_operation(self,frame):

        frame = self.apply_filters(frame)

        return frame



        
    
    
    
