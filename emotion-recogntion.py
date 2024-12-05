import cv2
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization
from facial_emotion_recognition import EmotionRecognition

# Define the DefaultConv2D helper function
def DefaultConv2d(filters, kernel_size=3, strides=1, padding="same", **kwargs):
    """A default convolution layer with commonly used settings."""
    return Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False, **kwargs)

# Define the ResidualUnit class
class ResidualUnit(Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2d(filters, strides=strides),
            BatchNormalization(),
            self.activation,
            DefaultConv2d(filters),
            BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2d(filters, kernel_size=1, strides=strides),
                BatchNormalization()
            ]

    def call(self, inputs):
        # Main path
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        # Skip connection
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        # Add skip connection and apply activation
        return self.activation(Z + skip_Z)

# Facial Emotion Recognition Function
def run_facial_emotion_recognition():
    er = EmotionRecognition(device='cpu')
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Press ESC to exit...")
    while True:
        success, frame = cam.read()
        if not success:
            print("Failed to capture frame. Exiting...")
            break

        frame = er.recognise_emotion(frame, return_type='BGR')
        cv2.imshow("Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            print("Exiting...")
            break

    cam.release()
    cv2.destroyAllWindows()

# Testing the Residual Unit
def test_residual_unit():
    residual_unit = ResidualUnit(filters=64, strides=2)
    input_tensor = tf.random.normal([1, 32, 32, 3])  # Batch of 1, 32x32 RGB images
    output_tensor = residual_unit(input_tensor)
    print("Residual Unit Test:")
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)

# Main Function
if __name__ == "__main__":
    print("1. Running Residual Unit Test")
    test_residual_unit()
    print("\n2. Starting Facial Emotion Recognition")
    run_facial_emotion_recognition()
