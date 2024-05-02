import datetime
import re  # Regular expression library

class ParkingRules:
    def __init__(self):
        self.current_date = datetime.datetime.now()
        self.day_name = self.current_date.strftime('%A')
        self.current_hour = self.current_date.hour

    def parking_rule(self, wordList):
        if len(wordList) == 1 and wordList[0] == 'P':
            return "Parking is permitted for 24 hours"
        elif 'P' in wordList and '&' in wordList:
            return "Parking is permitted for Handicap only, special permit needed"
        elif 'Avgift' in wordList:
            return self.handle_fee_parking(wordList)

    def handle_fee_parking(self, wordList):
        index = wordList.index('Avgift')
        # We look for a numeric entry or a time range close to 'Avgift'
        time_info = None
        for i in range(max(0, index - 2), min(len(wordList), index + 3)):  # Check two items around 'Avgift'
            if 'tim' in wordList[i]:
                time_info = self.extract_time_details(wordList, i)
            elif '-' in wordList[i]:
                if time_info:
                    time_info['interval'] = self.extract_interval(wordList[i])
                else:
                    time_info = {'interval': self.extract_interval(wordList[i])}
        
        if time_info:
            return self.calculate_parking_availability(**time_info)
        return "Check parking information on the sign."

    def extract_time_details(self, wordList, timeIndex):
        # Use regex to find the first sequence of digits in the string
        matches = re.findall(r'\d+', wordList[timeIndex])
        if matches:
            time = int(matches[0])  # Convert the first match to an integer
        else:
            time = None  # Handle case where no digits are found
        return {'time': time}

    def extract_interval(self, interval_str):
        # Use regex to find all sequences of digits in the string
        matches = re.findall(r'\d+', interval_str)
        if matches:
            # Convert all matches to integers
            return list(map(int, matches))
        else:
            return None

    def calculate_parking_availability(self, time, interval):
        if interval and len(interval) == 2:
            start_time, end_time = interval
            if self.day_name not in ['Saturday', 'Sunday'] and start_time <= self.current_hour < end_time:
                if (self.current_hour + time) <= end_time:
                    return f"Parking is permitted for max {time} hours with fee"
                else:
                    new_time = (self.current_hour + time) - end_time
                    if new_time < time:
                        return f"Parking is permitted for {time} hours with fee then it's free"
