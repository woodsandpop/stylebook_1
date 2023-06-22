import streamlit as st
import torch
import detect
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import time

def imageInput(src):
    
    if src == 'Upload your own PCB Image':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded PCB Image', use_column_width=True)
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            #call Model prediction--
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='pcb_1st/weights/best.pt', force_reload=True)  
            #model.cuda() if device == 'cuda' else model.cpu()
            pred = model(imgpath)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            #--Display predicton
            
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='AI PCB Predictions', use_column_width=True)

    elif src == 'From test PCB Images': 
        # Image selector slider
        imgpath = glob.glob('data/images/test/*')
        imgsel = st.slider('Select random images from test set.', min_value=1, max_value=len(imgpath), step=1) 
        image_file = imgpath[imgsel-1]
        submit = st.button("Predict PCB Defect!")
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col2:            
            if image_file is not None and submit:
                #call Model prediction--
                model = torch.hub.load('ultralytics/yolov5','custom', path= 'pcb_1st/weights/best.pt', force_reload=True) 
                pred = model(image_file)
                pred.render()  # render bbox in image
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(os.path.join('data/outputs', os.path.basename(image_file)))
                #--Display predicton
                    img_ = Image.open(os.path.join('data/outputs', os.path.basename(image_file)))
                    st.image(img_, caption='AI PCB Predictions')

def main():
    
    st.image("logo.JPG", width = 500)
    st.title("Coresys Limited")
    st.header("AI Tool for PCB Defect Detection")
    st.header("👈🏽 Select the Image Source options")
    st.sidebar.title('⚙️Options')
    src = st.sidebar.radio("Select input source.", ['From test PCB Images', 'Upload your own PCB Image'])
    imageInput(src)
   
if __name__ == '__main__':
    
    main()
    
@st.cache
def loadModel():
    start_dl = time.time()
    model_file = wget.download('https://archive.org/download/yoloTrained/yoloTrained.pt', out="models/")
    finished_dl = time.time()
    print(f"Model Downloaded, ETA:{finished_dl-start_dl}")
    
loadModel()
