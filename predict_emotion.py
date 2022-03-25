import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

root_dir = os.getcwd()
if (os.path.isdir(f'{os.getcwd()}/movieclassifier_new')):
    root_dir = f'{os.getcwd()}/movieclassifier_new'
print('before model instantiation')
emotion_model = load_model(f'{root_dir}/model.h5')
print(emotion_model)

def predict_emotion(file):
  # transform = transforms.ToTensor()
  # x = transform(photo)
  img = image.load_img(file, color_mode="grayscale", target_size=(48, 48))
  x = image.img_to_array(img)
  # print(x)

  x = np.expand_dims(x, axis = 0)
  x /= 255
  # print(x)
  print(x.shape)
  try:
    custom = emotion_model.predict(x)
  except:
    print('can not predict')
  print('after emotion prediction')
  # return emotion_analysis(custom[0])

def emotion_analysis(emotions):
  objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
  y_pos = np.arange(len(objects))
    
  plt.bar(y_pos, emotions, align='center', alpha=0.5)
  plt.xticks(y_pos, objects)
  plt.ylabel('percentage')
  plt.title('emotion')

  print('y_pos')
  print(objects[np.argmax(emotions)])
  # plt.show()
  # plt.savefig(f'foo{np.random.randint(0, 10000)}.png')
  return objects[np.argmax(emotions)]