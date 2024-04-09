from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
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

          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          contrast = cv2.convertScaleAbs(gray, alpha=2, beta=0)
          blur = cv2.GaussianBlur(contrast, (5, 5), 0)
          edged = cv2.Canny(blur, 30, 150)
          _, roi = cv2.threshold(edged, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU ) 
          contours, _ = cv2.findContours(roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

          for contour in contours:
              perimeter = cv2.arcLength(contour, True)
              approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
              if len(approx) == 4 and cv2.contourArea(contour) < 100:  
                  continue

              x, y, w, h = cv2.boundingRect(contour)
              # More refined aspect ratio check here
              if w < 50 or h < 20:  
                  continue

              roi = gray[y:y+h, x:x+w]
              text = pytesseract.image_to_string(roi, lang='swe', config='--psm 6')
              # Simple post-processing: Check if the result is likely to be valid
              if text.strip() not in self.recognized_words and len(text.strip()) > 3:
                    self.recognized_words.add(text.strip())
                    print(text)
                    cv2.imshow(f'ROI {x},{y}', roi)


    



    def on_stop(self):
        self.cap.release()

if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

    MainApp().run()
