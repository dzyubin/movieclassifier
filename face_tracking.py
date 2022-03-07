from facenet_pytorch import MTCNN
import mmcv, cv2
import torch
import numpy as np
import sys
from PIL import Image, ImageDraw
import os
from predict_emotion import predict_emotion

# from tensorflow.keras.preprocessing import image
# import torchvision.transforms as transforms
# from tensorflow.keras.models import load_model

print(torch.__version__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

def process_video(filename):
  # since MTCNN is a collection of neural nets and other code, the device must be passed in the following way to enable copying of objects when needed internally.
  mtcnn = MTCNN(keep_all=True, device=device)
  
  root_dir = os.getcwd()
  if (os.path.isdir(f'{os.getcwd()}/movieclassifier_new')):
    root_dir = f'{os.getcwd()}/movieclassifier_new'

  static_files_path = f'{root_dir}/static'
  filename, file_extension = os.path.splitext(filename)

  #loading a video with some faces in it. The mmcv PyPI package by mmlabs is used to read the video frames (it can be installed with pip install mmcv). Frames are then converted to PIL images
  # movieclassifier_new is the name of the root project directory on the hosting
  video = mmcv.VideoReader(f'{static_files_path}/untracked/{filename}{file_extension}')
  frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]

  # from IPython.display import HTML
  # from base64 import b64encode
  # mp4 = open('uploads/video.mp4','rb').read()
  # data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  # HTML("""
  # <video width=400 controls>
  #       <source src="%s" type="video/mp4">
  # </video>
  # """ % data_url)

  # iterate through each frame, detect faces, and draw their bounding boxes on the video frames.
  frames_tracked = []
  for i, frame in enumerate(frames):
    # if i < 35 or i > 70:
    if i < 35 or i > 40:
      continue
    print('\rTracking frame: {}'.format(i + 1), end='')
    # if i < 10:
      # print(frame)
    
    # Detect faces
    boxes, _ = mtcnn.detect(frame)
    # if i < 1:
    #   frame.save('frame.jpg')
    #   print(boxes)
    
    # Draw faces
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    if boxes is not None:
      for j, box in enumerate(boxes):
        print(box.tolist())
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
        cropped = frame.crop(box.tolist())
        cropped.save(f"cropped_{i}-{j}.jpg")
        predict_emotion(cropped)
        print(cropped)
    
    # Add to frame list
    frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
    print('\nDone')

  # Save tracked video
  dim = frames_tracked[0].size
  fourcc = cv2.VideoWriter_fourcc(*'FMP4')
  # fourcc = cv2.VideoWriter_fourcc(*'H264')
  # fourcc = cv2.VideoWriter_fourcc(*'avc1')
  # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
  # fourcc = cv2.VideoWriter_fourcc(*'PIM1')
  video_tracked = cv2.VideoWriter(f'{static_files_path}/tracked/{filename}_tracked.mp4', fourcc, 25.0, dim)
  for frame in frames_tracked:
    print(frame)
    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
  video_tracked.release()
