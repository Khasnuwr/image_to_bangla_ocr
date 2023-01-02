import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import torch
import easyocr
from PIL import ImageFont, ImageDraw, Image, ImageGrab
import numpy as np




def plate_detection():
    path = 'best.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)
    frame_window = st.image([])
    capture = cv2.VideoCapture(0)
    plateholder = st.empty()

    while True:
        success, frame = capture.read()

        if not success:
            break
        else:
            results = model(frame)
            reader = easyocr.Reader(['bn'], gpu=True)
            result = reader.readtext(frame)
            frame = np.squeeze(results.render())

            text = ''
            for string in result:
                text = text+string[1]+'\n'

            plateholder.empty()
            plateholder.text(text)

            # codes bellow commented for future development

            # fontpath = "banglamn.ttc"  # <== 这里是宋体路径
            #font = ImageFont.truetype(fontpath, 24)
            #img_pil = Image.fromarray(frame)
            #draw = ImageDraw.Draw(img_pil)
            #b, g, r, a = 221, 82, 6, 0
            #draw.text((50, 80),  text, font=font, fill=(b, g, r, a))
            #frame = np.array(img_pil)

            ###############################################

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            frame_window.image(frame)

            hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
            st.markdown(hide_streamlit_style, unsafe_allow_html=True)
     

st.title("Yolo Plate")
#plate_detection()


cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

class VideoProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")

		

		return av.VideoFrame.from_ndarray(frm, format='bgr24')


#st.title('SOMETHING WENT WRONG ;(')
webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
				rtc_configuration=RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
					)
	)
