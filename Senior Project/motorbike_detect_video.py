from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort
from collections import defaultdict

# Constants for file paths
VIDEO_PATH = "/Users/phasin/Project_withLane/Video_1.mp4"
OUTPUT_PATH = "output.mp4"

# Load YOLO models
motorcycle_model = YOLO("yolov8n.pt")  # General model for motorcycle detection
helmet_model = YOLO("bestV8_helmet_v2.pt")  # Custom model for helmet detection

# Initialize SORT tracker
mot_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Initialize counters and trackers
motorcycle_count = 0
helmet_count = 0
person_count = 0
tracked_objects = defaultdict(lambda: {'type': None, 'counted': False})

# Constants for detection confidence and size
PERSON_CONF_THRESHOLD = 0.4
MOTORCYCLE_CONF_THRESHOLD = 0.3
HELMET_CONF_THRESHOLD = 0.1
MIN_HELMET_SIZE = 0
MAX_HELMET_SIZE = 2000

# Video I/O Setup
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer
out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# Main processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run motorcycle and person detection
    results = motorcycle_model(frame, conf=MOTORCYCLE_CONF_THRESHOLD, classes=[0, 3])

    # Extract boxes from motorcycle/person detection
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    conf_scores = results[0].boxes.conf.cpu().numpy()

    # Prepare detections for tracking
    detections = []
    helmet_detections = []

    # Process person detections and run helmet detection in ROI
    for box, cls_id, conf in zip(boxes, classes, conf_scores):
        if cls_id == 0 and conf >= PERSON_CONF_THRESHOLD:
            detections.append([*box, conf, cls_id])
            
            # Create ROI for helmet detection
            x1, y1, x2, y2 = map(int, box)
            head_y2 = int(y1 + (y2-y1) * 0.4)  # Focus on upper body
            head_roi = frame[max(0, y1-20):head_y2, max(0, x1-20):min(x2+20, width)]
            
            if head_roi.size > 0:
                helmet_results_roi = helmet_model(head_roi, conf=HELMET_CONF_THRESHOLD)[0]
                
                for helmet_box in helmet_results_roi.boxes.xyxy.cpu().numpy():
                    hx1, hy1, hx2, hy2 = map(int, helmet_box)
                    
                    # Calculate absolute coordinates
                    abs_hx1 = max(0, x1-20) + hx1
                    abs_hy1 = max(0, y1-20) + hy1
                    abs_hx2 = max(0, x1-20) + hx2
                    abs_hy2 = max(0, y1-20) + hy2
                    
                    # Filter by size
                    helmet_width = abs_hx2 - abs_hx1
                    helmet_height = abs_hy2 - abs_hy1
                    if MIN_HELMET_SIZE <= helmet_width <= MAX_HELMET_SIZE and \
                       MIN_HELMET_SIZE <= helmet_height <= MAX_HELMET_SIZE:
                        helmet_detections.append([abs_hx1, abs_hy1, abs_hx2, abs_hy2, conf, 44])
        
        elif cls_id == 3 and conf >= MOTORCYCLE_CONF_THRESHOLD:
            detections.append([*box, conf, cls_id])

    # Add helmet detections to main detection list
    if helmet_detections:
        detections.extend(helmet_detections)

    # Update tracker
    if len(detections) > 0:
        tracked_objects_sort = mot_tracker.update(np.array(detections))
    else:
        tracked_objects_sort = np.empty((0, 6))

    # Process tracked objects
    for track in tracked_objects_sort:
        track_id = int(track[4])
        class_id = int(track[5])
        box = track[:4]
        
        # Count unique objects
        if not tracked_objects[track_id]['counted']:
            if class_id == 3:  # Motorcycle
                motorcycle_count += 1
                tracked_objects[track_id] = {'type': 'motorcycle', 'counted': True}
            elif class_id == 44:  # Helmet
                helmet_count += 1
                tracked_objects[track_id] = {'type': 'helmet', 'counted': True}
            elif class_id == 0:  # Person
                person_count += 1
                tracked_objects[track_id] = {'type': 'person', 'counted': True}

        # Set color and label based on class
        if class_id == 44:  # Helmet
            color = (0, 255, 0)  # Green
            label = f"Helmet #{track_id}"
            # Add helmet indicator circle
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            radius = int((box[2] - box[0]) / 4)
            cv2.circle(frame, (center_x, center_y), radius, color, 2)
        elif class_id == 3:  # Motorcycle
            color = (255, 0, 0)  # Blue
            label = f"Motorcycle #{track_id}"
        elif class_id == 0:  # Person
            color = (0, 0, 255)  # Red
            label = f"Person #{track_id}"
        
        # Draw detection box
        cv2.rectangle(frame, 
                     (int(box[0]), int(box[1])), 
                     (int(box[2]), int(box[3])), 
                     color, 2)
        
        # Add label with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(frame, 
                     (int(box[0]), int(box[1]) - label_height - baseline - 5),
                     (int(box[0]) + label_width, int(box[1])),
                     color, -1)
        
        cv2.putText(frame, label,
                   (int(box[0]), int(box[1]) - baseline - 5),
                   font, font_scale, (255, 255, 255), thickness)

    # Add count information
    cv2.putText(frame, f"Motorcycles: {motorcycle_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Helmets: {helmet_count}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Persons: {person_count}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Object Tracking", frame)
    
    # Write the frame to output video
    out.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()