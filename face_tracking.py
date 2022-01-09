from facenet_pytorch import MTCNN
import mmcv, cv2
import torch
import sys
from PIL import Image, ImageDraw
from IPython import display
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