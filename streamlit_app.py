import streamlit as st
from retinaface import RetinaFace
from streamlit_webrtc import webrtc_streamer
import av
import cv2


class VideoProcessor:
    def __init__(self) -> None:
        # RetinaFace class option
        self.quality = 'normal'
        # parameter for option change
        self.qualityChange = self.quality
        # face detector
        self.detector = RetinaFace(self.quality)

    def recv(self, frame):
        
        # if streamlit selectbox option is changed
        if self.qualityChange != self.quality:
            self.detector = RetinaFace(self.qualityChange)
            self.quality = self.qualityChange
        
        # get image from VideoFrame object and covert to ndarray
        rgb_image = frame.to_ndarray(format="rgb24")
        # get face detection info
        faces = self.detector.predict(rgb_image)
        # draw faces on image
        result_img = self.detector.draw(rgb_image, faces)
        # convert image to VideoFrame object and return
        return av.VideoFrame.from_ndarray(result_img, format="rgb24")


if __name__ == "__main__":
    
    st.title("Face Detection App")
    st.write("This is Face Detection App using RetinaFace Model")

    # Execute webRTC, Allocate VideoProcessor(We Made), setting STUN server for remote peers 
    ctx = webrtc_streamer(
        key="face_detection_app",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    if ctx.video_processor:
        ctx.video_processor.qualityChange = st.selectbox('model performance type',('speed','normal','original'))