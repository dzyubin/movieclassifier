from deepface import DeepFace
import cv2

def predict_emotion_deepface(file):
  print(file)
  # transform = transforms.ToTensor()
  # x = transform(photo)
#   img = image.load_img(file, color_mode="grayscale", target_size=(48, 48))
  # img = image.load_img(file)
  img = cv2.imread(file)
  print(img)
  try:
    predictions = DeepFace.analyze(img, enforce_detection=False)
  except Exception as inst:
    print(inst)

  return predictions