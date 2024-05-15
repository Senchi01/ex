import re
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
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

    def build(self):
      layout = MDBoxLayout(orientation='vertical')
      self.image = Image()
      self.capture_button = MDRaisedButton(
          text='Capture and Read',
          pos_hint={'center_x': 0.5, 'center_y': 0.5},
          size_hint=(None, None))
      self.capture_button.bind(on_press=self.takePicture)
      self.cap = cv2.VideoCapture(0)  
      Clock.schedule_interval(self.update, 1.0 / 30.0) 
      self.outputlabel = MDLabel(
          pos_hint={'center_x': 0.5, 'center_y': 0.5},
          size_hint=(0.5, None),
          font_size='20sp',
        ) 
      layout.add_widget(self.outputlabel)
      layout.add_widget(self.image)
      layout.add_widget(self.capture_button)

      return layout

    def update(self, dt):
      ret, frame = self.cap.read()
      if ret:
          buf = cv2.flip(frame, 0).tobytes()
          texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
          texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
          self.image.texture = texture

    def takePicture(self, *args):
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

          print(detected_texts)
          parkingRule = self.parkingRule(detected_texts)
          if parkingRule is not None:
            self.outputlabel.text = parkingRule    
          else:
            self.outputlabel.text = "No text are read"

    def parkingRule(self, wordList):
      current_date = datetime.datetime.now()
      dayName = current_date.strftime('%A')
      currentHour = current_date.hour
      output = ''
      if len(wordList) == 1 and wordList[0] == 'P':
        output = "Parking is permitted for 24 hours"
      elif 'Avgift' in wordList:
        avgiftIndex = wordList.index('Avgift')
        if 'tim' in wordList[avgiftIndex+1]: 
          elmt = wordList[avgiftIndex+1]
          time = self.extractTime(elmt)
          timeIndex = wordList.index(elmt)
          if dayName not in ['Saturday', 'Sunday']:
            output = self.handleFees(wordList,timeIndex+1,currentHour,time,isFee=True)
          else:
            output = self.handleWeekend(wordList, currentHour, time, dayName, isFee=True)
        elif 'tim' in wordList[avgiftIndex-1]:
          elmt = wordList[avgiftIndex-1]
          time = self.extractTime(elmt)
          timeIndex = wordList.index(elmt)
          if dayName not in ['Saturday', 'Sunday']:
            output = self.handleFees(wordList,avgiftIndex+1,currentHour,time,isFee=True)
          else:
            output = self.handleWeekend(wordList, currentHour, time, dayName,isFee=True)
      elif 'P-tillstånd' in wordList:
        output = f'Parking not permitted. permission needed!'
      else:
         for timeIndex, elmt in enumerate(wordList):
            if 'tim' in elmt:
              time = self.extractTime(elmt)
              if dayName not in ['Saturday', 'Sunday']:
                if len(wordList) > timeIndex + 1:
                  output = self.handleFees(wordList,timeIndex+1,currentHour,time,isFee=False)
                  break
                else:
                  output = f"parking is permitted for {time} hours with parking disc\nthen you need to park again."
              else:
                if len(wordList) > timeIndex + 1:
                  output = self.handleWeekend(wordList, currentHour, time, dayName,isFee=False)
                else:
                  output = f"parking is permitted for {time} hours with parking disc\nthen you need to park again."
            elif 'tim' not in elmt :
              if dayName not in ['Saturday', 'Sunday'] and len(wordList) >= 2:
                output = self.handleFees(wordList,1,currentHour,0,isFee=False)
              else:
                output = self.handleWeekend(wordList, currentHour, 0, dayName,isFee=False)
      return output
    
    def handleWeekend(self, wordList, currentHour, time, dayName, isFee):
      for i in wordList:
          if '(' in i:
              elmtIndex = wordList.index(i)
              if dayName == "Saturday":
                  return self.handleFees(wordList, elmtIndex, currentHour, time, isFee)
              elif dayName == "Sunday":
                  if len(wordList) >= elmtIndex + 1:
                    return self.handleFees(wordList, elmtIndex + 1, currentHour, time, isFee)
      return "Parking is free"

    def handleFees(self,wordList,timeIntervalIndex, ct, time, isFee ):
        output = ''
        timeInterval = wordList[timeIntervalIndex]
        timeInterval = re.sub(r'[^\d-]', '', timeInterval) 
        if '-' in timeInterval:
          timeSplitted = timeInterval.split('-')
          startTime = int(timeSplitted[0])
          if startTime in ['B',88, '&']:
              startTime = 8
          endTime = int(timeSplitted[1])
        else:
           return timeInterval
        if time != 0:
          if startTime < ct and ct < endTime:
            if (ct + int(time)) <= endTime and isFee :
              output = f"Parking is permitted for {time} hours with fee"
            elif (ct + int(time)) > endTime and isFee:
              newTime = (ct + int(time)) - endTime
              if newTime < int(time) :
                output = f"Parking is permitted for {newTime} hours with fee then it's free"
            elif (ct + int(time)) <= endTime and not isFee:
               output = f"Parking is permitted for {time} hours for free then you need to park again"
            elif (ct + int(time)) > endTime and not isFee:  
              output = f"Parking is permitted"
               
          elif len(wordList) > timeIntervalIndex + 1 and ct < startTime:
            output = f"Parking is free to {startTime}"
          else:
            output = f"Parking is permitted"
        else:
          if startTime < ct and ct < endTime:
            output = f"parking is permitted until {endTime}, then no parking"
          else:
            output = f"Parking is not permitted"
        return output
                      

    def extractTime(self, element):
      time_numbers = re.findall(r'\d+', element)
      if time_numbers:
          return int(time_numbers[0])
      else:
          return 0 
    
    def onStop(self):
        self.cap.release()

if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

    MainApp().run()
