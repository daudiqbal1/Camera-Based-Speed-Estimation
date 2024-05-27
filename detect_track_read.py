from ultralytics import YOLO
import cv2
import pytesseract
import csv
import torch

# Initialize the OCR engine
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  # Adjust this path as necessary

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Load video
video_path = './test_5.mp4'
cap = cv2.VideoCapture(video_path)

ret = True

# Data structures for tracking
id_to_plate = {}
plate_to_id = {}

# Open a CSV file to store the number plates
with open('detected_plates.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'License Plate'])

    # Read frames
    frame_count = 0
    while ret:
        ret, frame = cap.read()
        frame_count += 1

        if ret:
            # Detect and track objects
            results = model.track(frame, persist=True)  # persist helps to remember the tracked object

            # Process results and extract text
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Extract the coordinates from the tensor
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy

                    # Extract the tracking ID and ensure it is not None
                    track_id = box.id
                    if track_id is None:
                        continue
                    track_id = int(track_id)

                    # Extract the region of interest (ROI)
                    roi = frame[y1:y2, x1:x2]

                    # Use pytesseract to extract text
                    plate_text = pytesseract.image_to_string(roi, config='--psm 8')
                    plate_text = plate_text.strip()

                    # Update tracking data structures
                    if plate_text:
                        if track_id in id_to_plate:
                            plate_to_id.pop(id_to_plate[track_id], None)
                        id_to_plate[track_id] = plate_text
                        plate_to_id[plate_text] = track_id

                        # Save detected plate to CSV
                        writer.writerow([frame_count, plate_text])
                        print(f'Frame {frame_count}: Detected license plate - {plate_text}')
                    
                    # Display the license plate text in the bounding box
                    display_text = id_to_plate.get(track_id, "Unknown")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Visualize
            cv2.imshow('frame', frame)
            # Press q to exit the visualization
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
