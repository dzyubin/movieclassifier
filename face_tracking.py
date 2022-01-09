from facenet_pytorch import MTCNN
import mmcv, cv2
import torch
import sys
from PIL import Image, ImageDraw
from IPython import display
# print("Python version")
# print (sys.version)
# print("Version info.")
# print (sys.version_info)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))