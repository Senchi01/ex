import re
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np
import pytesseract
import datetime 

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


    def parking_rule(self, wordList):
      current_date = datetime.datetime.now()
      dayName = current_date.strftime('%A')
      currentHour = current_date.hour

      if len(wordList) == 1 and wordList[0] == 'P':
        return "Parking is permitted for 24 hours"
      elif wordList[0] == 'P' and wordList[1] == '&':
        return "Parking is permitted for Handicap only"
      elif 'Avgift' in wordList:
        avgiftIndex = wordList.index('Avgift')
        if 'tim' in wordList[avgiftIndex+1]: 
          elmt = wordList[avgiftIndex+1]
          time = elmt[0]
          timeIndex = wordList.index(elmt)
          if dayName not in ['Saturday', 'Sunday']:
            return self.handleFees(wordList,timeIndex,currentHour,time,isFee=True)
          elif dayName == 'Saturday':
              for i in wordList:
                 if '(' in i:
                    elmtIndex = wordList.index(i)
                    return self.handleFees(wordList,elmtIndex-1,currentHour,time,isFee=True)  
                 else:
                    return f"parking is free"  
          else:
            for i in wordList:
              if '(' in i:
                elmtIndex = wordList.index(i)
                return self.handleFees(wordList,elmtIndex-1,currentHour,time,isFee=True)
              else:
                    return f"parking is free"  
        elif 'tim' in wordList[avgiftIndex-1]:
          elmt = wordList[avgiftIndex-1]
          time = elmt[0]
          timeIndex = wordList.index(elmt)
          if dayName not in ['Saturday', 'Sunday']:
            return self.handleFees(wordList,avgiftIndex,currentHour,time,isFee=True)
          elif dayName == 'Saturday':
              for i in wordList:
                 if '(' in i:
                    elmtIndex = wordList.index(i)
                    return self.handleFees(wordList,elmtIndex-1,currentHour,time,isFee=True) 
                 else:
                    return f"parking is free"    
          else:
            for i in wordList:
              if '(' in i:
                elmtIndex = wordList.index(i)
                return self.handleFees(wordList,elmtIndex-1,currentHour,time,isFee=True)
              else:
                    return f"parking is free"  
      elif 'P-tillstånd' in wordList:
        return f'Parking not permitted. permission needed!'
      elif 'Avgift' not in wordList:
        for elmt in wordList:
           if 'tim' in elmt:
              time = elmt[0]
              timeIndex = wordList.index(elmt)
              return self.handleFees(wordList,timeIndex,currentHour,time,isFee=False)


       


    def handleFees(self,wordList,index, ct, time, isFee ):
        if len(wordList) > index + 1:
           index += 1
        else:
           index = index
        timeInterval = wordList[index]
        timeIntervalIndex = wordList.index(timeInterval)
        timeInterval = re.sub(r'[^\d-]', '', timeInterval) 
        timeSplitted = timeInterval.split('-')
        startTime = int(timeSplitted[0])
        if startTime in ['B',88, '&']:
            startTime = 8
        endTime = int(timeSplitted[1])
         # Convert `time` by extracting digits and turning into an integer
        timeDigits = re.findall(r'\d+', time)  # Finds all digit groups
        if timeDigits:
          time = int(timeDigits[0])  # Take the first group of digits found
        if startTime < ct and ct < endTime:
          if (ct + int(time)) <= endTime and isFee:
            return f"Parking is permitted for max {time} hours with fee"
          elif (ct + int(time)) > endTime:
            newTime = (ct + int(time)) - endTime
            if newTime < int(time) and isFee :
              return f"Parking is permitted for {newTime} hours with fee then it's free"
            else:
              return f"Parking is permitted for free"
          else:
            return f"Parking is permitted for free"
        elif len(wordList) > timeIntervalIndex+1 and ct < startTime:
          return f"Parking is free to {startTime}"
        else:
          return f"Parking is permitted"
                

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
                  roi = masked[y:y+h, x:x+w]
                  # Convert ROI to grayscale
                  roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                  # Apply a binary threshold to the ROI
                  _, roi_thresh = cv2.threshold(roi_gray, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                  # Adjusting pytesseract config
                  custom_config=r'--oem 3 --psm 6'
                  
                  text=pytesseract.image_to_string(roi_thresh,
                                                   lang='swe',
                                                   config=custom_config).strip()

                  if text:
                    detected_texts.extend(text.split('\n'))

          # Sort the detected text areas based on the y-coordinate
          # Output the text in order
          for text in detected_texts:
            print(text)
          print(self.parking_rule(detected_texts))
          print(detected_texts)
    
    def on_stop(self):
        self.cap.release()

if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

    MainApp().run()
