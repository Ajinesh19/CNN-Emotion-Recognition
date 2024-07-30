from facial_emotion_recognition import EmotionRecognition
import cv2
er =EmotionRecognition (device='cpu')
cam =cv2.VideoCapture (1) 
while True: 
  success,frame cam.read() 
  print (sucess)
  frame= er.recognise_emotion(frame, return_type-'BGR') 
  cv2.imshow("Frame", frame) 
  keycv2.waitKey(1) 
  print(key)
  if key=-27:
      break
car.release()
cv2.destroyAllWindows ()
