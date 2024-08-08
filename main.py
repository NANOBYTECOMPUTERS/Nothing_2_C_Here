import bettercam
from PIL import Image
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import configparser
import logging
import time
import win32api
import win32con

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class PIDController:
    def __init__(self, kp, ki, kd, prediction_time=5, integral_limit=200, deadband=1.0, output_limit=10.0):
        """
        Initializes a PID controller with the given parameters.

        Args:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
            prediction_time (float): Time in seconds for predicting future target position.
            integral_limit (float): Limit on the integral term to prevent windup.
            deadband (float): Region around the setpoint where no control action is taken.
            output_limit (float): Limit on the output of the PID controller.
        """

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_x = 0
        self.integral_y = 0
        self.last_error_x = 0
        self.last_error_y = 0
        self.last_time = time.time()
        self.prediction_time = prediction_time
        self.integral_limit = integral_limit
        self.deadband = deadband
        self.output_limit = output_limit
        self.prev_predicted_x = 0
        self.prev_predicted_y = 0

    def update_parameters(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def predict_target_position(self, target_x, target_y, prev_x, prev_y):
        velocity_x = target_x - prev_x
        velocity_y = target_y - prev_y
        predicted_x = target_x + velocity_x * self.prediction_time
        predicted_y = target_y + velocity_y * self.prediction_time
    
        if abs(predicted_x - self.prev_predicted_x) < 10 and abs(predicted_y - self.prev_predicted_y) < 10:
            predicted_x = self.prev_predicted_x + velocity_x * self.prediction_time
            predicted_y = self.prev_predicted_y + velocity_y * self.prediction_time
        self.prev_predicted_x = predicted_x
        self.prev_predicted_y = predicted_y
        return predicted_x, predicted_y

    def calculate_movement(self, target_x, target_y, center_x, center_y):

        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        if dt == 0:
            dt = 1e-6 
        error_x = target_x - center_x
        error_y = target_y - center_y

        if abs(error_x) < self.deadband:
            error_x = 0
        if abs(error_y) < self.deadband:
            error_y = 0
        # Apply PID
        proportional_x = self.kp * error_x
        proportional_y = self.kp * error_y

        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        self.integral_x = np.clip(self.integral_x, -self.integral_limit, self.integral_limit)
        self.integral_y = np.clip(self.integral_y, -self.integral_limit, self.integral_limit)
        integral_x = self.ki * self.integral_x
        integral_y = self.ki * self.integral_y

        derivative_x = self.kd * (error_x - self.last_error_x) / dt
        derivative_y = self.kd * (error_y - self.last_error_y) / dt
        self.last_error_x = error_x
        self.last_error_y = error_y

        move_x = proportional_x + integral_x + derivative_x
        move_y = proportional_y + integral_y + derivative_y

        # Limit Output
        move_x = np.clip(move_x, -self.output_limit, self.output_limit)
        move_y = np.clip(move_y, -self.output_limit, self.output_limit)

        return move_x, move_y


        
def load_config(config_path='box_config.ini'):
    config = configparser.ConfigParser()
    config.read(config_path)
    try:
        target_size = config.getint('BoxDimensions', 'size')
    except (configparser.NoSectionError, configparser.NoOptionError):
        logging.warning("Config section or option missing, using default size.")
        target_size = 320
    logging.info(f"Target size set to {target_size}")
    return target_size

@torch.inference_mode()
def perform_detection(model, image):
    logging.info("Performing detection")
    return model.predict(
        source=image,
        cfg="game.yaml",  # Make sure this path is correct
        imgsz=640,
        stream=True,
        conf=0.3,
        iou=0.5,
        device=0,
        half=True,
        max_det=20,
        agnostic_nms=False,
        augment=False,
        vid_stride=False,
        visualize=False,
        verbose=False,
        show_boxes=False,
        show_labels=False,
        show_conf=False,
        save=False,
        show=False
    )

def distance_from_center(x, y, center_x, center_y):
    return np.sqrt((x - center_x)**2 + (y - center_y)**2)

def main():
    logging.info("Starting main function")
    target_size = load_config()
    image_width = 1920
    image_height = 1080

    logging.info("Initializing camera")
    camera = bettercam.create(device_idx=0, output_idx=0, output_color="BGR")

    # region of interest
    left, top = (image_width - target_size) // 2, (image_height - target_size) // 2
    right, bottom = left + target_size, top + target_size
    region = (left, top, right, bottom)

    # Initialize tracking
    track_history = defaultdict(lambda: [])
    model = YOLO("9.pt", task="segment")
    model.to("cuda")
    names = model.model.names
    temp_id_counter = 0
        # PID Controller Setup
    pid_controller = PIDController(kp=0.5, ki=0.01, kd=0.1, prediction_time=5)  # Adjust values
    last_locked_position = None
    center_x, center_y = target_size // 2, target_size // 2
    target_fps = 60
    max_track_length = 3

    while True:
        frame = camera.grab(region=region)
        if frame is None:
            logging.warning("Frame grab failed, continuing")
            continue

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and boxes.data.shape[0] > 0 and hasattr(boxes, 'id') and boxes.id is not None:
                boxes = boxes.cpu()
                xywh = boxes.xywh
                conf = boxes.conf
                cls = boxes.cls
                track_ids = [int(track_id) for track_id in boxes.id.tolist()]

                for i, track_id in enumerate(track_ids):
                    if track_id == -1:
                        temp_id_counter += 1
                        track_ids[i] = temp_id_counter

                closest_head_distance = float('inf')
                closest_player_distance = float('inf')
                locked_track_id = None

                for (x, y, w, h), track_id, cls_id, confidence in zip(xywh, track_ids, cls, conf):
                    center_x_box, center_y_box = int(x), int(y)
                    distance = distance_from_center(center_x_box, center_y_box, center_x, center_y)

                    if names[int(cls_id)] == 'head' and distance < closest_head_distance:
                        closest_head_distance = distance
                        locked_track_id = track_id
                    elif names[int(cls_id)] == 'player' and distance < closest_player_distance:
                        closest_player_distance = distance
                        if locked_track_id is None:
                            locked_track_id = track_id

                for track_id in track_ids:
                    if track_id not in track_history:
                        track_history[track_id] = []
                
                for (x, y, w, h), track_id, cls_id, confidence in zip(xywh, track_ids, cls, conf):
                    x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                    box_color = (0, 165, 255) if track_id == locked_track_id else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    label = f"{names[int(cls_id)]} : {track_id} ({confidence:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                    track = track_history[track_id]
                    track.append((float(x), float(y)))

                    if len(track) > max_track_length:
                        track.pop(0)

                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    line_color = (0, 255, 0) if track_id == locked_track_id else (37, 255, 225)
                    cv2.polylines(frame, [points], isClosed=False, color=line_color, thickness=2)

                    cv2.circle(frame, (int(track[-1][0]), int(track[-1][1])), 5, (235, 219, 11), -1)
                prev_locked_x = 0
                prev_locked_y = 0

                if locked_track_id is not None:  # Only adjust if a track is locked
                    locked_x, locked_y = track_history[locked_track_id][-1]

                    # Predict Target Position
                    if last_locked_position is not None:
                        predicted_x, predicted_y = pid_controller.predict_target_position(
                            locked_x, locked_y, prev_locked_x, prev_locked_y 
                        )
                    else:
                        predicted_x, predicted_y = locked_x, locked_y
                    prev_locked_x = locked_x
                    prev_locked_y = locked_y
                    last_locked_position = (locked_x, locked_y)


                    # Calculate PID-based Movement
                    x, y = pid_controller.calculate_movement(
                        predicted_x, predicted_y, center_x, center_y
                    )
                    win32api.mouse_event (int(x), int(y), 0, 0)
        cv2.imshow("detections", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            logging.info("Quit signal received, exiting")
            break

    camera.release()
    cv2.destroyAllWindows()
    logging.info("Resources released, application terminated")
    


if __name__ == "__main__":
    main()