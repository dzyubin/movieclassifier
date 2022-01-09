from facenet_pytorch import MTCNN
import mmcv, cv2
import torch
import numpy as np
import sys
from PIL import Image, ImageDraw
# from IPython import display
import os
# print("Python version")
# print (sys.version)
# print("Version info.")
# print (sys.version_info)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# since MTCNN is a collection of neural nets and other code, the device must be passed in the following way to enable copying of objects when needed internally.
mtcnn = MTCNN(keep_all=True, device='cpu')

print(os.listdir())
#loading a video with some faces in it. The mmcv PyPI package by mmlabs is used to read the video frames (it can be installed with pip install mmcv). Frames are then converted to PIL images
video = mmcv.VideoReader('uploads/video.mp4')
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
fourcc = cv2.VideoWriter_fourcc(*'FMP4')    
video_tracked = cv2.VideoWriter('uploads/video_tracked.mp4', fourcc, 25.0, dim)
for frame in frames_tracked:
  print(frame)
  video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
video_tracked.release()
