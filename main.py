from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np
import pytesseract

class MainApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recognized_words = set()
    def build(self):
        layout = MDBoxLayout(orientation='vertical')
        self.image = Image()
        layout.add_widget(self.image)
        self.capture_button = MDRaisedButton(
            text='Capture and Read',
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            size_hint=(None, None))
        self.capture_button.bind(on_press=self.take_picture)
        layout.add_widget(self.capture_button)
        self.cap = cv2.VideoCapture(0)  
        Clock.schedule_interval(self.update, 1.0 / 30.0)  
        return layout

    def update(self, dt):
        # Capture frame-by-frame
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to texture
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture


    def take_picture(self, *args):
      ret, frame = self.cap.read()
      if ret:
          # Convert to HSV color space to detect blue color
          hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
          # Define the range for blue color and create a mask
          lower_blue = np.array([100, 150, 50])
          upper_blue = np.array([140, 255, 255])
          mask = cv2.inRange(hsv, lower_blue, upper_blue)

          # Find contours in the mask
          contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

          for contour in contours:
              # Approximate the contour to a polygon
              epsilon = 0.1 * cv2.arcLength(contour, True)
              approx = cv2.approxPolyDP(contour, epsilon, True)

              # Check if the approximated polygon has four sides (i.e., is a rectangle or square)
              if len(approx) == 4:
                  x, y, w, h = cv2.boundingRect(approx)

                  # Extract ROI with some padding
                  padding = 5  # Adjust padding as needed
                  roi = frame[y-padding:y+h+padding, x-padding:x+w+padding]

                  # Convert ROI to grayscale and apply threshold
                  roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                  _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                  # Apply OCR on the thresholded ROI
                  text = pytesseract.image_to_string(roi_thresh, lang='swe', config='--psm 6')

                  # Simple post-processing: Check if the result is likely to be valid
                  if text.strip() and text.strip() not in self.recognized_words and len(text.strip()) > 3:
                      self.recognized_words.add(text.strip())
                      print(text)
                      # Display the ROI for debugging
                      cv2.imshow(f'ROI {x},{y}', roi_thresh)




    



    def on_stop(self):
        self.cap.release()

if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

    MainApp().run()
