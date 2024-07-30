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

class ResidualUnit(tf.keras.layers.Layer):
  def __init__(self, filters, strides=1, activation="relu", **kwargs):
    super().__init__(**kwargs)
    self.activation = tf.keras.activations.get(activation)
    self.main_layers = [
        DefaultConv2d(filters, strides=strides),
        tf.keras.layers.BatchNormalization(),
        self.activation,
        DefaultConv2d(filters),
        tf.keras.layers.BatchNormalization()
    ]

    self.skip_layers = []
    if strides > 1:
      self.skip_layers = [
          DefaultConv2d(filters, kernel_size=1, strides=strides),
          tf.keras.layers.BatchNormalization()
      ]

      def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
          Z = layer(Z)

        skip_Z = inputs
        for layer in self.skip_layers:
          skip_Z = layer(skip_Z)

        return self.activation(Z + skip_Z)