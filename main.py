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
          lower_blue = np.array([90, 100, 50]) 
          upper_blue = np.array([159, 255, 255])  
          mask = cv2.inRange(hsv, lower_blue, upper_blue)
          masked = cv2.bitwise_and(frame, frame, mask=mask)

          # Find contours in the mask
          contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          
          # Sort contours based on the y-coordinate of their bounding rect, top to bottom
          sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
          # List to hold all detected text areas
          detected_texts = []

          for contour in sorted_contours:
              # Approximate the contour to a polygon
              epsilon = 0.1 * cv2.arcLength(contour, True)
              approx = cv2.approxPolyDP(contour, epsilon, True)

              # Check if the approximated polygon has four sides
              if len(approx) == 4:
                  x, y, w, h = cv2.boundingRect(approx)
    
                  # Extract ROI
                  roi = frame[y:y+h, x:x+w]
                  # Convert ROI to grayscale
                  roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                  # Apply a binary threshold to the ROI
                  _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                  # Adjusting pytesseract config
                  custom_config=r'--oem 3 --psm 6'
                  
                  text=pytesseract.image_to_string(roi_thresh,
                                                   lang='swe',
                                                   config=custom_config).strip()

                  if text:
                      detected_texts.append((y, text))

          # Sort the detected text areas based on the y-coordinate
          detected_texts.sort(key=lambda item: item[0])

          # Output the text in order
          for _,text in detected_texts:
              print(text)

    def on_stop(self):
        self.cap.release()

if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

    MainApp().run()
