import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
from collections import deque
import numpy as np
import imutils

    
class VideoTransformer(VideoTransformerBase):
    st.title('Object Detection & Tracking using OpenCV')
    st.title('A Project for Cloud Computing')
    buffer=64
    colorLower = (90, 100, 100)
    colorUpper = (110, 255, 255)
    pts = deque(maxlen=buffer)

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = imutils.resize(image, width=600)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.colorLower, self.colorUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        if(len(cnts) > 0):
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if radius > 5:
                cv2.circle(image, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv2.circle(image, center, 5, (0, 0, 255), -1)
            self.pts.appendleft(center)

        for i in range(1, len(self.pts)):
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue

            thickness = int(np.sqrt(self.buffer / float(i + 1)) * 2.5)
            cv2.line(image, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)

        return image
    

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
st.title('By :  Mihir Shah & Ujjaval Parmar')
