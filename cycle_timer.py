import cv2
import time
import argparse
import numpy as np
import csv
import tkinter as tk
from tkinter import simpledialog
from ultralytics import YOLO

# Function to pop up a naming box
def get_zone_name(zone_number):
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    name = simpledialog.askstring("ROI Setup", f"Enter name for Zone {zone_number}:")
    root.destroy()
    return name if name else f"Zone_{zone_number}"

def get_overlap_ratio(box, roi):
    xA, yA = max(box[0], roi[0]), max(box[1], roi[1])
    xB, yB = min(box[2], roi[2]), min(box[3], roi[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxArea = (box[2] - box[0]) * (box[3] - box[1])
    return interArea / float(boxArea) if boxArea > 0 else 0

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="robot_work.mp4")
parser.add_argument("--model", type=str, default="best.pt")
parser.add_argument("--visual", action="store_true")
parser.add_argument("--skip", type=int, default=3)
parser.add_argument("--robots", type=int, default=1)
parser.add_argument("--output", type=str, default="cycle_log.csv")
args = parser.parse_args()

model = YOLO(args.model).to('cuda')
cap = cv2.VideoCapture(args.source)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
time_inc = (1.0 / fps) * args.skip

ret, frame = cap.read()
if not ret: print("Video load failed"); exit()

# --- ZONE SETUP WITH POP-UP ---
zones = []
print("\n[SETUP] Draw ROI and look for the naming pop-up box.")
while True:
    r = cv2.selectROI("Define Zones - ESC when finished", frame, fromCenter=False)
    if r[2] == 0 or r[3] == 0: break
    
    # Trigger the tkinter pop-up
    name = get_zone_name(len(zones) + 1)
    
    zones.append({"name": name, "rect": (int(r[0]), int(r[1]), int(r[0]+r[2]), int(r[1]+r[3]))})
cv2.destroyAllWindows()

if len(zones) >= 1:
    CYCLE_SEQUENCE = [z["name"] for z in zones]
    if len(zones) > 1:
        CYCLE_SEQUENCE.append(zones[0]["name"]) 
    print(f"TRACKING SEQUENCE: {' -> '.join(CYCLE_SEQUENCE)}")
else:
    print("No zones defined. Exiting."); exit()

tracking_data = {}  
id_map = {}         
cycle_history = []
primary_robot_id = None 

# Run tracking
results = model.track(source=args.source, persist=True, tracker="bytetrack.yaml", stream=True, verbose=False, imgsz=640)



with open(args.output, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Cycle", "Tool_ID", "Duration", "End_Time"])

    frame_count = 0
    for result in results:
        frame_count += 1
        if frame_count % args.skip != 0: continue
        
        video_time = frame_count / fps
        display_frame = result.orig_img.copy() if args.visual else None
        active_zones_this_frame = set()

        if result.boxes.id is not None:
            boxes_all = result.boxes.xyxy.cpu().numpy()
            ids_all = result.boxes.id.int().cpu().numpy()
            conf_all = result.boxes.conf.cpu().numpy()

            if args.robots == 1:
                best_idx = np.argmax(conf_all)
                boxes, raw_ids = [boxes_all[best_idx]], [ids_all[best_idx]]
            else:
                boxes, raw_ids = boxes_all, ids_all

            for box, r_id in zip(boxes, raw_ids):
                if r_id not in id_map:
                    if args.robots == 1:
                        if primary_robot_id is None: primary_robot_id = r_id
                        id_map[r_id] = primary_robot_id
                    else:
                        id_map[r_id] = r_id
                
                tid = id_map[r_id]
                if tid not in tracking_data:
                    tracking_data[tid] = {'next_step': 0, 'start_time': None, 'last_triggered_zone': None}

                for z in zones:
                    overlap = get_overlap_ratio(box, z['rect'])
                    if overlap >= 0.1: # 10% overlap threshold
                        active_zones_this_frame.add(z['name'])
                        
                        if z['name'] != tracking_data[tid]['last_triggered_zone']:
                            expected = CYCLE_SEQUENCE[tracking_data[tid]['next_step']]
                            
                            if z['name'] == expected:
                                if tracking_data[tid]['next_step'] == 0:
                                    tracking_data[tid]['start_time'] = video_time
                                
                                tracking_data[tid]['next_step'] += 1
                                tracking_data[tid]['last_triggered_zone'] = z['name']
                                print(f" [STEP] Tool {tid} -> {z['name']}")

                                if tracking_data[tid]['next_step'] >= len(CYCLE_SEQUENCE):
                                    duration = video_time - tracking_data[tid]['start_time']
                                    if duration > 0.5: # Filter out noise
                                        cycle_history.append(duration)
                                        writer.writerow([len(cycle_history), tid, round(duration, 2), round(video_time, 2)])
                                        print(f" >>> [CYCLE COMPLETE] {duration:.2f}s")
                                    tracking_data[tid]['next_step'] = 0

                if args.visual:
                    bx = box.astype(int)
                    cv2.rectangle(display_frame, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 255), 2)
                    cv2.putText(display_frame, f"ROBOT {tid}", (bx[0], bx[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if args.visual:
            for z in zones:
                is_active = z['name'] in active_zones_this_frame
                color = (0, 255, 0) if is_active else (255, 0, 0)
                cv2.rectangle(display_frame, (z['rect'][0], z['rect'][1]), (z['rect'][2], z['rect'][3]), color, 2 if is_active else 1)
                cv2.putText(display_frame, z['name'], (z['rect'][0], z['rect'][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imshow("Robot Monitoring System", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

print(f"\nFinished. {len(cycle_history)} cycles logged.")
cap.release()
cv2.destroyAllWindows()