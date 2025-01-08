import time
from datetime import datetime

#make time eazier to read
class Ptime():
    def __init__(self):
        self.saved_time = ""
        self.saved_datetime = ""
        self.month_number_dict = {
        'Jan': '01',
        'Feb': '02',
        'Mar': '03',
        'Apr': '04',
        'May': '05',
        'Jun': '06',
        'Jul': '07',
        'Aug': '08',
        'Sep': '09',
        'Oct': '10',
        'Nov': '11',
        'Dec': '12'
        }
    def set_time_now(self):
        self.saved_time = str(time.ctime())
        self.saved_datetime = datetime.now()
        
    def get_origin_time(self):
        return self.saved_time
        
    def get_time(self):
        time_list = self.saved_time.split(' ')
        if '' in time_list:
            time_list.remove('')
        if(int(time_list[2]) < 10):
            time_list[2] = "0" + time_list[2]
        time_list[1] = self.month_number_dict[time_list[1]]
        mask = [4, 1, 2, 3]
        ptime = ""
        for i in mask:
            ptime += time_list[i]
        return ptime
        
    def get_time_to_hour(self):
        hour_time = self.saved_datetime.strftime("%Y_%m_%d_%H")
        return str(hour_time)
    
    def get_time_to_minute(self):
        minute_time = self.saved_datetime.strftime("%Y_%m_%d_%H_%M")
        return str(minute_time)
        
if __name__ == "__main__":
    t = Ptime()
    t.set_time_now()
    print("time now: ", t.get_time())
    print("time to hour now: ", t.get_time_to_hour())
    print("time to minute now: ", t.get_time_to_minute())
