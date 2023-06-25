import streamlit as st
import tempfile
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import cv2
import random

# st.set_page_config(page_title="Car accident Detection", page_icon=":sos:")

# Load your deep learning model
def load_model():
    # model = YOLO('n1.pt')
    model = YOLO('m2.pt')
#     model = YOLO('x1.pt')
    return model

# Define a function to process the uploaded video
def process_video(video_file):
    # Load your model
    model = load_model()
    
    video = cv2.VideoCapture(video_file)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    output_path = 'cv2output.mp4'
    result = cv2.VideoWriter(output_path, fourcc, fps, size)

    ret, frame = video.read()
    while ret:
        res = model.predict(source=frame, save=False, conf=0.5)
        print(res[0])
        res_img = res[0].plot()
        result.write(res_img)
        ret, frame = video.read()

    video.release()
    result.release()
    
    return True

# Define Streamlit app
def main():
    st.set_page_config(page_title="Car accident Detection", page_icon=":sos:")
    st.title("Deep Learning Video Processing")
    
    # File upload section
    st.header("Upload Video")
    video_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    
    if video_file is not None:
        # Create a temporary directory to store the uploaded video
        temp_dir = tempfile.mkdtemp()
        
        # Save the uploaded video to the temporary directory
        video_path = os.path.join(temp_dir, video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        
        
        # Process the video
        if st.button("Process Video"):
            with st.spinner(text='In progress'):
            	output_video = process_video(video_path)
            	st.success('Detection finished; saving video...')
            	output_path = random.randint(0, 1000)
            	os.system(f'ffmpeg -i cv2output.mp4 -vcodec libx264 output/{output_path}.mp4')
            
            if output_video == True:
            	# Display the processed video
            	while not os.path.isfile(f'output/{output_path}.mp4'):
            		pass
            	st.header("Processed Video")
            	video_bytes = open(f'output/{output_path}.mp4', 'rb').read()
            	st.video(video_bytes)
        
    
    # Sidebar credits
    st.sidebar.info("This is a demo app for deep learning video processing to detect accidents in roads with YOLOv8 model.")
    st.sidebar.write('**Developed by**')
    st.sidebar.write('*Tarek Ashraf Mahmoud*')
    st.sidebar.write('*Osama Anter Afifi*')
    st.sidebar.write('*Ahmed Mohamed Ali*')
    st.sidebar.write('*Ahmed Mohamed Ibrahim*')
    st.sidebar.write('*Adham Mohamed Tawfik*')
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.write('**Under Supervision**')
    st.sidebar.write('*Dr. Ayat Mohammed*')
    st.sidebar.write('*TA. Marwa Shams*')
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.write('Sponsored by **Ain Shams University**')
    
if __name__ == "__main__":
    main()
