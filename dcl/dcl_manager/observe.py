import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class Target:
    
    watchDir = "/home/pi/dcl/dcl_manager/models"
    
    def __init__(self):
        self.observer = Observer()
        
    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.watchDir, recursive=True)
        self.observer.start()

        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Error")
        self.observer.join()

global sum
sum =0

class Handler(FileSystemEventHandler):
#FileSystemEventHandler 클래스를 상속받음.
#아래 핸들러들을 오버라이드 함
    @staticmethod
    def on_any_event(event):
        global sum
        if event.is_directory==True:
            sum +=999

        if event.event_type == "created":
            sum +=1

            print("Received created event - %s" % event.src_path)
            filename = event.src_path.split("/")[-1]
            dirname = event.src_path.split("/")[-2]
            print(filename, dirname)

        if sum ==1000:
            sum=0
            print("here")
        print(sum)

if __name__ =="__main__": #본 파일에서 실행될 때만 실행되도록 함
    
    w = Target()
    w.run()
