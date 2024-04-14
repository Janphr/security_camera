from time import time
import cv2 as cv
from ultralytics import YOLO
from collections import deque
import threading
from pytapo import Tapo
import math
import datetime
import subprocess as sp
import shlex
import os


ip = "192.168.0.105"
port = 554
# Login from Tapo App: Device Settings -> Advanced Settings -> Camera Account
username = ""
password = ""
# Password of Tapo App
cloudPassword = ""

# To prevent unwanted recordings. Static, frequently detected objects will be ignored automatically
ignore_class_list = ['bench', 'fire hydrant', 'chair', 'giraffe', 'cow', 'bench', 'chair', 'potted plant', 'couch']
# We only want to chase away cats!
alarm_class_list = ['cat']

class VideoCapture:
  rec_jobs = {}  
  data_path = "./data/"
  def __init__(self, src):
    if not os.path.exists(self.data_path):
        os.makedirs(self.data_path)
        
    self.cap = cv.VideoCapture(src)
    self.q = deque(maxlen=60)

    t = threading.Thread(target=self._reader, daemon=True)
    t.start()

  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      self.q.append(frame)
      for rec_job in self.rec_jobs.values():
        if 'thread' in rec_job: continue
        rec_job['frames'].append(frame)

  def read(self):
    return self.q[-1]

  def start_recording(self, filename_appendix):
    for rec_job in self.rec_jobs.values():
        if 'thread' not in rec_job: return
        
    rec_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if filename_appendix:
        rec_filename += "_" + str(filename_appendix)
    self.rec_jobs[rec_filename] = {
        'frames': list(self.q)
    }
    print(f"Recording {rec_filename}")

  def _stop_recording(self, frames, filename, use_ffmpeg=False):        
    
        fps = self.cap.get(cv.CAP_PROP_FPS)
        width, height = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        output_filename = self.data_path + filename + ".mp4"
        
        if use_ffmpeg:        
            process = sp.Popen(shlex.split(f'ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx265 -pix_fmt yuv420p -crf 24 {output_filename}'), stdin=sp.PIPE)
            for frame in frames:
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            process.wait()
            process.terminate()
        else:   
            fourcc = cv.VideoWriter_fourcc(*'mp4v') # 'mp4v'
            out = cv.VideoWriter(filename=output_filename, fourcc=fourcc, fps=fps, frameSize=(width, height), isColor=True)
            for frame in frames:
                out.write(frame)
            out.release()
        del self.rec_jobs[filename]

  def stop_recording(self):
    for rec_filename, rec_job in self.rec_jobs.items():
        if 'thread' in rec_job: continue
        print(f"Stopping recording {rec_filename}")
        rec_job['thread'] = threading.Thread(target=self._stop_recording, args=(rec_job['frames'], rec_filename), daemon=True).start()
    
def calculate_centroid(box):
    # Calculate centroid of a rectangle
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2

def calculate_distance(coord1, coord2):
    # Calculate distance between two points
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf, verbose=False)
    else:
        results = chosen_model.predict(img, conf=conf, verbose=False)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    detections = []
    for result in results:
        for box in result.boxes:
            cv.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
            detections.append((result.names[int(box.cls[0])], time(), calculate_centroid(box.xyxy[0])))
    return img, detections

def show(frame):
    cv.imshow("Security Camera", frame)
    cv.waitKey(1)
    
class ObjectMemory:
    def __init__(self, tags):
        self.detections = {tag: {} for tag in tags}
        self.rem_duration = 10
        self.centroid_similarity_threshold = 50
        
    def check(self, detections):
        now = time()
        # Remove detections older than rem_duration
        for tag, dets in self.detections.items():
            if dets:
                dets[:] = [det for det in dets if now - det['last_seen'] < self.rem_duration]
        
        new_detections = []
        for tag, t, c in detections:
            # no detections for this tag, add
            if len(self.detections[tag]) == 0:
                self.detections[tag] = [{'last_seen': t, 'centroid': c}]
                new_detections.append((tag, t, c))
                continue
            
            # determine detections that are farther than centroid_similarity_threshold
            not_similar_dets = [det for det in self.detections[tag] if calculate_distance(c, det['centroid']) > self.centroid_similarity_threshold]
            # if all detections are not similar, add to new detections
            if len(not_similar_dets) == len(self.detections[tag]):
                new_detections.append((tag, t, c))
                
            self.detections[tag] = not_similar_dets
            self.detections[tag].append({'last_seen': t, 'centroid': c})

                    
        return new_detections
                    

def main():
    
    model = YOLO("yolov9c.pt")
    
    om = ObjectMemory(list(model.names.values()))
    
    cap = VideoCapture(f"rtsp://{username}:{password}@{ip}:{port}/stream1")
    
    tapo = Tapo(ip, 'admin', cloudPassword, cloudPassword)

    print(tapo.getBasicInfo())
      
    alarm = False
    
    frames_without_detections = 0
    wait_frames = 5
    
    try:
        while True:
            frame = cap.read()
            
            result_frame, detections = predict_and_detect(model, frame, classes=[], conf=0.75)
            detections = om.check(detections)
            
            if detections:
                rec_class_name = next((x[0] for x in detections if x[0] not in ignore_class_list), None)
                alarm_class_name = next((x[0] for x in detections if x[0] in alarm_class_list), None)
                
                if rec_class_name:
                    frames_without_detections = 0
                    cap.start_recording(rec_class_name)
                else:
                    frames_without_detections += 1
                    if frames_without_detections > wait_frames:
                        cap.stop_recording()
                
                if alarm_class_name:
                    if not alarm:
                        tapo.startManualAlarm()
                        alarm = True
                        print(f"Alarm set due to {alarm_class_name} detection...")
                elif alarm:
                    tapo.stopManualAlarm()
                    alarm = False
                    print("Alarm stopped...")
            else:
                frames_without_detections += 1
                if frames_without_detections > wait_frames:
                    cap.stop_recording()
                elif alarm:
                    tapo.stopManualAlarm()
                    alarm = False
                    print("Alarm stopped...")
                
            show(result_frame)

    except KeyboardInterrupt:
        pass

    cv.destroyAllWindows()

    pass

if __name__ == "__main__":
    main()
