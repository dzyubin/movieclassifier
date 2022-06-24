from facenet_pytorch import MTCNN
import mmcv, cv2
import torch
import numpy as np
import sys
from PIL import Image, ImageDraw, ImageFont
import os
# from predict_emotion import predict_emotion
from predict_emotion_deepface import predict_emotion_deepface

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
  print("before 'video' file is created")
  video = mmcv.VideoReader(f'{static_files_path}/untracked/{filename}{file_extension}')
  print("before frames array created")
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
  # print(frames)
  for i, frame in enumerate(frames):
    # if i < 5 or i > 46:
      # continue
    print('\rTracking frame: {}'.format(i + 1), end='')
    
    # Detect faces
    boxes, _ = mtcnn.detect(frame)
    
    # Draw faces
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    # print(frame)
    # print(boxes)
    if boxes is not None:
      for j, box in enumerate(boxes):
        # print(box)
        face_bounding_box = box.tolist()
        draw.rectangle(face_bounding_box, outline=(255, 0, 0), width=6)

        face_img_filename = f"{root_dir}\cropped_{i}-{j}.jpg"
        face_img = frame.crop(face_bounding_box)
        face_img.save(face_img_filename)
        emotion_predictions = predict_emotion_deepface(face_img_filename)
        # emotion_predictions = predict_emotion_deepface(f'{root_dir}/angry_download.jpg')
        print(emotion_predictions)

        print(face_img_filename)

        try:
          os.remove(face_img_filename)
        except:
          print('Image file doesn\'t exist')

        font_size = 45
        # font = ImageFont.truetype("static/arial.ttf", font_size)
        font = ImageFont.truetype(f"{static_files_path}/arial.ttf", font_size)
        # font = ImageFont.load_default(font_size)
        # font = ImageFont.load_default()
        # TODO: increase font size
        draw.text((
          face_bounding_box[0] + face_bounding_box[2] / 2,
          face_bounding_box[1] + face_bounding_box[3] / 2),
          emotion_predictions['dominant_emotion'],
          'lime',
          font=font
        )

    # Add to frame list
    # TODO: set width/height dynamically
    # frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
    frames_tracked.append(frame_draw.resize((640, 960), Image.BILINEAR))
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
    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
  video_tracked.release()

# TODO: finish refactoring
def draw_bounding_box():
  face_bounding_box = box.tolist()
  draw.rectangle(face_bounding_box, outline=(255, 0, 0), width=6)

  face_img_filename = f"cropped_{i}-{j}.jpg"
  face_img = frame.crop(face_bounding_box)
  face_img.save(face_img_filename)
  # emotion_predictions = predict_emotion_deepface(f'{root_dir}/angry_download.jpg')
  emotion_predictions = predict_emotion_deepface(face_img_filename)
  print(emotion_predictions)

  # TODO: delete image !!!!!!!!!!!!!!!!!!!!!

  font_size = 45
  font = ImageFont.truetype("arial.ttf", font_size)
  draw.text((
    face_bounding_box[0] + face_bounding_box[2] / 2,
    face_bounding_box[1] + face_bounding_box[3] / 2),
    emotion_predictions['dominant_emotion'],
    'lime',
    font=font
  )
