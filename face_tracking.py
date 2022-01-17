from facenet_pytorch import MTCNN
import mmcv, cv2
import torch
import numpy as np
import sys
from PIL import Image, ImageDraw
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

def process_video(filename):
  # since MTCNN is a collection of neural nets and other code, the device must be passed in the following way to enable copying of objects when needed internally.
  mtcnn = MTCNN(keep_all=True, device=device)
  
  root_dir = os.getcwd()
  if (os.path.isdir(f'{os.getcwd()}/movieclassifier_new')):
    root_dir = f'{os.getcwd()}/movieclassifier_new'

  static_files_path = f'{root_dir}/static'
  print(static_files_path)
  file_extension = filename.split('.')[1]
  print(file_extension)

  #loading a video with some faces in it. The mmcv PyPI package by mmlabs is used to read the video frames (it can be installed with pip install mmcv). Frames are then converted to PIL images
  # movieclassifier_new is the name of the root project directory on the hosting
  # try:
  video = mmcv.VideoReader(f'{static_files_path}/untracked/{filename}')
  # except:
    # video = mmcv.VideoReader(f'{os.getcwd()}/static/untracked/{filename}')
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
    print('\rTracking frame: {}'.format(i + 1), end='')
    
    # Detect faces
    boxes, _ = mtcnn.detect(frame)
    
    # Draw faces
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    if boxes is not None:
      for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    
    # Add to frame list
    frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
    print('\nDone')

  # Save tracked video
  dim = frames_tracked[0].size
  # fourcc = cv2.VideoWriter_fourcc(*'FMP4')
  fourcc = cv2.VideoWriter_fourcc(*'H264')
  # if (os.path.isdir(static_files_path)):
  video_tracked = cv2.VideoWriter(f'{static_files_path}/tracked/video_tracked.mp4', fourcc, 25.0, dim)
  # else:
    # video_tracked = cv2.VideoWriter(f'{os.getcwd()}/static/video_tracked.mp4', fourcc, 25.0, dim)
  for frame in frames_tracked:
    print(frame)
    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
  video_tracked.release()
