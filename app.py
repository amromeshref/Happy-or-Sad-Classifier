# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture

# Import other dependencies
import cv2
import tensorflow as tf

# Build app and layout 
class CamApp(App):

    def build(self):
        # Main layout components 
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text = "Check", on_press = self.verify, size_hint=(1,.1))
        self.check_label = Label(text="Checking is Uninitiated", size_hint=(1,.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.check_label)

        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model("my data/model.h5")

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout

    # Run continuously to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def image_transformation(self,img):
        # resizing the image to (1,100, 100, 3)
        img = cv2.resize(img,(100,100))
        img = img.reshape((1,100,100,3))
        # Normalizing the image
        img = img/255
        
        return img

    def verify(self, *args):

        # Getting the input image
        ret, frame = self.capture.read()

        # Preprocessing the input image
        input_image = self.image_transformation(frame)
        
         # Predicting
        y_hat = self.model.predict(input_image)
        y_hat = tf.nn.sigmoid(y_hat)
        happy = False
        if(y_hat[0] >= 0.5):
            happy = True
        
        self.check_label.text = "You are HAPPY!" if happy else "You are SAD"

if __name__ == '__main__':
    CamApp().run()
