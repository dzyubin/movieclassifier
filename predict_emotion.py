import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

root_dir = os.getcwd()
if (os.path.isdir(f'{os.getcwd()}/movieclassifier_new')):
    root_dir = f'{os.getcwd()}/movieclassifier_new'

emotion_model = load_model(f'{root_dir}/model.h5')

def predict_emotion(photo):
  print('sdf')
  # transform = transforms.ToTensor()
  # x = transform(photo)
  x = image.img_to_array(photo)

  x = np.expand_dims(x, axis = 0)
  x /= 255
  print(x)
  print(x.shape)
  custom = emotion_model.predict(x)
  emotion_analysis(custom[0])

def emotion_analysis(emotions):
  objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
  y_pos = np.arange(len(objects))
    
  plt.bar(y_pos, emotions, align='center', alpha=0.5)
  plt.xticks(y_pos, objects)
  plt.ylabel('percentage')
  plt.title('emotion')
    
  # plt.show()
  plt.savefig('foo.png')