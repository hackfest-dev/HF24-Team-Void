import cv2
import numpy as np
import os
import math
from frame_operations import FrameOperations

class PoseEstimator():
    def __init__(self):
        self.FRAME_OPS = FrameOperations()

        self.BODY_PARTS =  { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
            
        self.POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

        self.CWD = os.getcwd()
        self.RESOURCES = os.path.join(self.CWD,'resources')
        self.GRAPH_OPT = os.path.join(self.RESOURCES,'graph_opt.pb')

        self.NET = cv2.dnn.readNetFromTensorflow(self.GRAPH_OPT)
        self.THR = 0.3
        self.IN_WIDTH = 367
        self.IN_HEIGHT = 248

        self.POINTS = []

        self.KEY_DISTANCES = {"RArm":{"RShoulder-RElbow":None,"RElbow-RWrist":None,"Neck-RShoulder":None},
        "LArm":{"LShoulder-LElbow":None,"LElbow-LWrist":None,"Neck-LShoulder":None},
        "RLeg":{"RHip-RKnee":None,"RKnee-RAnkle":None},
        "LLeg":{"LHip-RKnee":None,"LKnee-RAnkle":None}}

        self.KEY_ANGLES = {"RArm": [],"LArm":[],"RLeg":[],"LLeg":[]}

        self.TEXT_COLOR = (0,0,0)

    def rad_to_deg(self,rad):
        return rad * (180/math.pi)

    def get_pose_key_angles_filtered(self, frame, wantBlank=False):
        # Define indices of joints to keep
        joints_to_keep = [4, 7, 3, 10, 9, 8, 13]  # Index of joints: wrist, elbow, shoulder, foot, knee, hip, foot

        # Initialize a blank image to overlay the filtered red dots
        blank_frame = np.zeros_like(frame)

        # Loop through each joint
        for i, (joint_index, point) in enumerate(zip(self.BODY_PARTS.values(), self.POINTS)):
            # Check if the joint is among those to keep
            if i in joints_to_keep and point is not None:
                # Draw a red dot at the joint position
                cv2.ellipse(blank_frame, point, (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        if wantBlank:
            return blank_frame

        # Overlay the filtered red dots on the original frame
        result_frame = cv2.addWeighted(frame, 1, blank_frame, 0.7, 0)

        return result_frame

    def get_pose_key_angles(self, frame, wantBlank=False):
    # Define the key points positions
        RShoulder_pos = None
        RWrist_pos = None

        LShoulder_pos = None
        LWrist_pos = None

        Neck_pos = None
        
        RElbow_pos = None
        LElbow_pos = None

        RHip_pos = None
        RKnee_pos = None
        RAnkle_pos = None

        LHip_pos = None
        LKnee_pos = None
        LAnkle_pos = None

        frame_h, frame_w = frame.shape[:2]
            
        self.NET.setInput(cv2.dnn.blobFromImage(frame, 1.0, (self.IN_WIDTH, self.IN_HEIGHT), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = self.NET.forward()

        out = out[:, :19, :, :]

        assert(len(self.BODY_PARTS) == out.shape[1])

        # Clear to get new points
        self.POINTS.clear()

        for i in range(len(self.BODY_PARTS)):
            
            heatMap = out[0, i, :, :]
            
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frame_w * point[0]) / out.shape[3]
            y = (frame_h * point[1]) / out.shape[2]

            if(conf > self.THR):
                self.POINTS.append((int(x), int(y)))
            else:
                self.POINTS.append(None)

        if wantBlank:
            return self.get_pose_key_angles_filtered(frame, wantBlank=True)

        for pair in self.POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in self.BODY_PARTS)
            assert(partTo in self.BODY_PARTS)

            idFrom = self.BODY_PARTS[partFrom]
            idTo = self.BODY_PARTS[partTo]

            # if found points (if not found, returns None)
            if self.POINTS[idFrom] and self.POINTS[idTo]:

                # Assign positions for each key point
                if partFrom == "RShoulder":
                    RShoulder_pos = self.POINTS[idFrom]
                elif partTo == "RWrist":
                    RWrist_pos = self.POINTS[idTo]
                elif partFrom == "LShoulder":
                    LShoulder_pos = self.POINTS[idFrom]
                elif partTo == "LWrist":
                    LWrist_pos = self.POINTS[idTo]
                elif partFrom == "Neck":
                    Neck_pos = self.POINTS[idFrom]
                elif partTo == "RElbow":
                    RElbow_pos = self.POINTS[idTo]
                elif partTo == "LElbow":
                    LElbow_pos = self.POINTS[idTo]
                elif partFrom == "RHip":
                    RHip_pos = self.POINTS[idFrom]
                elif partTo == "RKnee":
                    RKnee_pos = self.POINTS[idTo]
                elif partTo == "RAnkle":
                    RAnkle_pos = self.POINTS[idTo]
                elif partFrom == "LHip":
                    LHip_pos = self.POINTS[idFrom]
                elif partTo == "LKnee":
                    LKnee_pos = self.POINTS[idTo]
                elif partTo == "LAnkle":
                    LAnkle_pos = self.POINTS[idTo]

                # Calculate distances between key points
                if partFrom == "RShoulder" and partTo == "RElbow":
                    self.KEY_DISTANCES["RArm"]["RShoulder-RElbow"] = ((self.POINTS[idFrom][0] - self.POINTS[idTo][0]) ** 2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) ** 2) ** 0.5
                elif partFrom == "RElbow" and partTo == "RWrist":
                    self.KEY_DISTANCES["RArm"]["RElbow-RWrist"] = ((self.POINTS[idFrom][0] - self.POINTS[idTo][0]) ** 2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) ** 2) ** 0.5
                elif partFrom == "LShoulder" and partTo == "LElbow":
                    self.KEY_DISTANCES["LArm"]["LShoulder-LElbow"] = ((self.POINTS[idFrom][0] - self.POINTS[idTo][0]) ** 2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) ** 2) ** 0.5
                elif partFrom == "LElbow" and partTo == "LWrist":
                    self.KEY_DISTANCES["LArm"]["LElbow-LWrist"] = ((self.POINTS[idFrom][0] - self.POINTS[idTo][0]) ** 2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) ** 2) ** 0.5
                elif partFrom == "Neck" and partTo == "RShoulder":
                    self.KEY_DISTANCES["RArm"]["Neck-RShoulder"] = ((self.POINTS[idFrom][0] - self.POINTS[idTo][0]) ** 2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) ** 2) ** 0.5
                elif partFrom == "Neck" and partTo == "LShoulder":
                    self.KEY_DISTANCES["LArm"]["Neck-LShoulder"] = ((self.POINTS[idFrom][0] - self.POINTS[idTo][0]) ** 2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) ** 2) ** 0.5
                elif partFrom == "RHip" and partTo == "RKnee":
                    self.KEY_DISTANCES["RLeg"]["RHip-RKnee"] = ((self.POINTS[idFrom][0] - self.POINTS[idTo][0]) ** 2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) ** 2) ** 0.5
                elif partFrom == "RKnee" and partTo == "RAnkle":
                    self.KEY_DISTANCES["RLeg"]["RKnee-RAnkle"] = ((self.POINTS[idFrom][0] - self.POINTS[idTo][0]) ** 2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) ** 2) ** 0.5
                elif partFrom == "LHip" and partTo == "LKnee":
                    self.KEY_DISTANCES["LLeg"]["LHip-LKnee"] = ((self.POINTS[idFrom][0] - self.POINTS[idTo][0]) ** 2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) ** 2) ** 0.5
                elif partFrom == "LKnee" and partTo == "LAnkle":
                    self.KEY_DISTANCES["LLeg"]["LKnee-LAnkle"] = ((self.POINTS[idFrom][0] - self.POINTS[idTo][0]) ** 2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) ** 2) ** 0.5

                # Draw lines and points on the frame
                cv2.line(frame, self.POINTS[idFrom], self.POINTS[idTo], (255, 0, 0), 2)
                cv2.ellipse(frame, self.POINTS[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, self.POINTS[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        # Calculate and display the clenching percentages
        for limb, distances in self.KEY_DISTANCES.items():
            if limb == "LArm":
                # Calculate the angle between LShoulder, LElbow, and LWrist
                if "LShoulder-LElbow" in distances and "LElbow-LWrist" in distances:
                    if self.POINTS[self.BODY_PARTS["LShoulder"]] is not None and self.POINTS[self.BODY_PARTS["LElbow"]] is not None and self.POINTS[self.BODY_PARTS["LWrist"]] is not None:
                        a = distances["LShoulder-LElbow"]
                        b = distances["LElbow-LWrist"]
                        c = ((self.POINTS[self.BODY_PARTS["LShoulder"]][0] - self.POINTS[self.BODY_PARTS["LWrist"]][0]) ** 2 + (self.POINTS[self.BODY_PARTS["LShoulder"]][1] - self.POINTS[self.BODY_PARTS["LWrist"]][1]) ** 2) ** 0.5
                        if a != 0 and b != 0:
                            try:
                                angle = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
                                percentage = 100 - ((angle / math.pi) * 100)
                                if not math.isnan(percentage):
                                    print(f"{limb}: {percentage:.2f}%")
                                    cv2.putText(frame, f"{limb}: {percentage:.2f}%", (self.POINTS[self.BODY_PARTS["LElbow"]][0], self.POINTS[self.BODY_PARTS["LElbow"]][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            except ValueError as e:
                                print(f"Error calculating angle for {limb}: {e}")
            elif limb == "RArm":
                # Calculate the angle between RShoulder, RElbow, and RWrist
                if "RShoulder-RElbow" in distances and "RElbow-RWrist" in distances:
                    if self.POINTS[self.BODY_PARTS["RShoulder"]] is not None and self.POINTS[self.BODY_PARTS["RElbow"]] is not None and self.POINTS[self.BODY_PARTS["RWrist"]] is not None:
                        a = distances["RShoulder-RElbow"]
                        b = distances["RElbow-RWrist"]
                        c = ((self.POINTS[self.BODY_PARTS["RShoulder"]][0] - self.POINTS[self.BODY_PARTS["RWrist"]][0]) ** 2 + (self.POINTS[self.BODY_PARTS["RShoulder"]][1] - self.POINTS[self.BODY_PARTS["RWrist"]][1]) ** 2) ** 0.5
                        if a != 0 and b != 0:    
                            try:
                                angle = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
                                percentage = 100 - ((angle / math.pi) * 100)
                                if not math.isnan(percentage):
                                    print(f"{limb}: {percentage:.2f}%")
                                    cv2.putText(frame, f"{limb}: {percentage:.2f}%", (self.POINTS[self.BODY_PARTS["RElbow"]][0], self.POINTS[self.BODY_PARTS["RElbow"]][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            except ValueError as e:
                                print(f"Error calculating angle for {limb}: {e}")
            elif limb == "LLeg":
                # Calculate the angle between LHip, LKnee, and LAnkle
                if "LHip-LKnee" in distances and "LKnee-LAnkle" in distances:
                    if self.POINTS[self.BODY_PARTS["LHip"]] is not None and self.POINTS[self.BODY_PARTS["LKnee"]] is not None and self.POINTS[self.BODY_PARTS["LAnkle"]] is not None:
                        a = distances["LHip-LKnee"]
                        b = distances["LKnee-LAnkle"]
                        c = ((self.POINTS[self.BODY_PARTS["LHip"]][0] - self.POINTS[self.BODY_PARTS["LAnkle"]][0]) ** 2 + (self.POINTS[self.BODY_PARTS["LHip"]][1] - self.POINTS[self.BODY_PARTS["LAnkle"]][1]) ** 2) ** 0.5
                        if a != 0 and b != 0:
                            try:
                                angle = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
                                percentage = 100 - ((angle / math.pi) * 100)
                                if not math.isnan(percentage):
                                    print(f"{limb}: {percentage:.2f}%")
                                    cv2.putText(frame, f"{limb}: {percentage:.2f}%", (self.POINTS[self.BODY_PARTS["LKnee"]][0], self.POINTS[self.BODY_PARTS["LKnee"]][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            except ValueError as e:
                                print(f"Error calculating angle for {limb}: {e}")
            elif limb == "RLeg":
                # Calculate the angle between RHip, RKnee, and RAnkle
                if "RHip-RKnee" in distances and "RKnee-RAnkle" in distances:
                    if self.POINTS[self.BODY_PARTS["RHip"]] is not None and self.POINTS[self.BODY_PARTS["RKnee"]] is not None and self.POINTS[self.BODY_PARTS["RAnkle"]] is not None:
                        a = distances["RHip-RKnee"]
                        b = distances["RKnee-RAnkle"]
                        c = ((self.POINTS[self.BODY_PARTS["RHip"]][0] - self.POINTS[self.BODY_PARTS["RAnkle"]][0]) ** 2 + (self.POINTS[self.BODY_PARTS["RHip"]][1] - self.POINTS[self.BODY_PARTS["RAnkle"]][1]) ** 2) ** 0.5
                        if a != 0 and b != 0:
                            try:
                                angle = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
                                percentage = 100 - ((angle / math.pi) * 100)
                                if not math.isnan(percentage):
                                    print(f"{limb}: {percentage:.2f}%")
                                    cv2.putText(frame, f"{limb}: {percentage:.2f}%", (self.POINTS[self.BODY_PARTS["RKnee"]][0], self.POINTS[self.BODY_PARTS["RKnee"]][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            except ValueError as e:
                                print(f"Error calculating angle for {limb}: {e}")


        return frame