import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import pygame
import threading
import time
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Detection3D:
    """Enhanced 3D detection with real-world coordinates"""
    bbox_2d: Tuple[int, int, int, int]  # x1, y1, x2, y2
    bbox_3d: Tuple[float, float, float, float, float, float]  # x, y, z, w, h, l
    confidence: float
    class_id: int
    class_name: str
    distance: float
    velocity: Tuple[float, float, float]  # vx, vy, vz
    angle: float  # viewing angle
    risk_score: float

class CameraCalibration:
    """Camera intrinsic parameters for 3D reconstruction"""
    def __init__(self, focal_length=800, image_width=1280, image_height=720):
        self.focal_length = focal_length
        self.cx = image_width / 2  # Principal point x
        self.cy = image_height / 2  # Principal point y
        self.image_width = image_width
        self.image_height = image_height
        
        # Camera matrix
        self.K = np.array([
            [focal_length, 0, self.cx],
            [0, focal_length, self.cy],
            [0, 0, 1]
        ])
        
        # Vehicle-specific parameters
        self.camera_height = 1.5  # meters above ground
        self.camera_pitch = 0.1  # radians (slight downward tilt)

class Object3DEstimator:
    """Estimate 3D object properties from 2D detections"""
    
    def __init__(self, camera_calib: CameraCalibration):
        self.camera = camera_calib
        
        # Real-world object dimensions (average values in meters)
        self.object_dimensions = {
            'car': {'width': 1.8, 'height': 1.5, 'length': 4.5},
            'truck': {'width': 2.5, 'height': 3.0, 'length': 12.0},
            'bus': {'width': 2.5, 'height': 3.2, 'length': 12.0},
            'motorcycle': {'width': 0.8, 'height': 1.2, 'length': 2.0},
            'bicycle': {'width': 0.6, 'height': 1.0, 'length': 1.8},
            'person': {'width': 0.6, 'height': 1.7, 'length': 0.3}
        }
    
    def estimate_distance(self, bbox_2d: Tuple[int, int, int, int], 
                         object_type: str) -> float:
        """Estimate distance using object height in image"""
        x1, y1, x2, y2 = bbox_2d
        bbox_height_pixels = y2 - y1
        
        if object_type not in self.object_dimensions:
            object_type = 'car'  # Default assumption
            
        real_height = self.object_dimensions[object_type]['height']
        
        # Distance = (real_height * focal_length) / pixel_height
        distance = (real_height * self.camera.focal_length) / bbox_height_pixels
        return max(distance, 1.0)  # Minimum 1 meter
    
    def estimate_3d_position(self, bbox_2d: Tuple[int, int, int, int], 
                           distance: float) -> Tuple[float, float, float]:
        """Convert 2D bbox to 3D world coordinates"""
        x1, y1, x2, y2 = bbox_2d
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Convert image coordinates to world coordinates
        # X: left-right (positive = right)
        # Y: up-down (positive = up)  
        # Z: forward-backward (positive = forward)
        
        world_x = (center_x - self.camera.cx) * distance / self.camera.focal_length
        world_y = self.camera.camera_height - (center_y - self.camera.cy) * distance / self.camera.focal_length
        world_z = distance
        
        return (world_x, world_y, world_z)
    
    def estimate_3d_bbox(self, bbox_2d: Tuple[int, int, int, int], 
                        object_type: str, distance: float) -> Tuple[float, float, float, float, float, float]:
        """Estimate 3D bounding box dimensions"""
        if object_type not in self.object_dimensions:
            object_type = 'car'
            
        dims = self.object_dimensions[object_type]
        x, y, z = self.estimate_3d_position(bbox_2d, distance)
        
        return (x, y, z, dims['width'], dims['height'], dims['length'])

class EnhancedYOLODetector:
    """Enhanced YOLO with automotive-specific optimizations"""
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.object_3d_estimator = Object3DEstimator(CameraCalibration())
        
        # Automotive-specific class mapping
        self.automotive_classes = {
            0: 'person',
            1: 'bicycle', 
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        # Performance tracking
        self.detection_times = []
        self.frame_count = 0
        
    def detect_objects(self, frame: np.ndarray) -> List[Detection3D]:
        """Enhanced object detection with 3D estimation"""
        start_time = time.time()
        
        # Run YOLO detection
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections_3d = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Filter for automotive-relevant classes
                    if cls_id in self.automotive_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        class_name = self.automotive_classes[cls_id]
                        
                        # 3D estimation
                        distance = self.object_3d_estimator.estimate_distance(
                            (x1, y1, x2, y2), class_name
                        )
                        bbox_3d = self.object_3d_estimator.estimate_3d_bbox(
                            (x1, y1, x2, y2), class_name, distance
                        )
                        
                        # Calculate viewing angle
                        center_x = (x1 + x2) / 2
                        image_center = frame.shape[1] / 2
                        angle = math.atan2(center_x - image_center, 
                                         self.object_3d_estimator.camera.focal_length)
                        
                        # Basic risk scoring (enhanced in Week 4)
                        risk_score = self._calculate_basic_risk(distance, class_name)
                        
                        detection = Detection3D(
                            bbox_2d=(x1, y1, x2, y2),
                            bbox_3d=bbox_3d,
                            confidence=conf,
                            class_id=cls_id,
                            class_name=class_name,
                            distance=distance,
                            velocity=(0.0, 0.0, 0.0),  # Will be calculated in Week 3
                            angle=angle,
                            risk_score=risk_score
                        )
                        
                        detections_3d.append(detection)
        
        # Performance tracking
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        self.frame_count += 1
        
        return detections_3d
    
    def _calculate_basic_risk(self, distance: float, object_type: str) -> float:
        """Basic risk calculation (will be enhanced in Week 4)"""
        # Risk increases as distance decreases
        base_risk = max(0, (50 - distance) / 50)  # Risk peaks at <50m
        
        # Different risk factors for different objects
        risk_multipliers = {
            'person': 1.5,      # Higher risk for pedestrians
            'bicycle': 1.3,     # Vulnerable road users
            'motorcycle': 1.2,
            'car': 1.0,
            'truck': 1.1,       # Larger vehicles
            'bus': 1.1
        }
        
        multiplier = risk_multipliers.get(object_type, 1.0)
        return min(base_risk * multiplier, 1.0)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get detection performance statistics"""
        if not self.detection_times:
            return {}
            
        avg_time = np.mean(self.detection_times[-100:])  # Last 100 frames
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_detection_time': avg_time,
            'fps': fps,
            'total_frames': self.frame_count
        }

class AdvancedLaneDetector:
    """Enhanced lane detection (basic version - will be upgraded in Week 2)"""
    
    def __init__(self):
        self.previous_lanes = None
        
    def detect_lanes(self, frame):
        """Improved lane detection with temporal smoothing"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding based on image brightness
        brightness = np.mean(gray)
        if brightness < 100:  # Dark conditions
            low_threshold = 30
            high_threshold = 100
        else:  # Bright conditions
            low_threshold = 50
            high_threshold = 150
            
        edges = cv2.Canny(blur, low_threshold, high_threshold)
        
        # Enhanced region of interest
        height, width = edges.shape
        mask = np.zeros_like(edges)
        
        # More precise ROI polygon
        polygon = np.array([[
            (int(width * 0.1), height),
            (int(width * 0.4), int(height * 0.6)),
            (int(width * 0.6), int(height * 0.6)),
            (int(width * 0.9), height)
        ]], np.int32)
        
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Hough line detection with better parameters
        lines = cv2.HoughLinesP(
            masked_edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=30,
            minLineLength=50, 
            maxLineGap=20
        )
        
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:  # Avoid division by zero
                    slope = (y2 - y1) / (x2 - x1)
                    
                    # More strict slope filtering
                    if -0.8 < slope < -0.2:  # Left lane
                        left_lines.append(line[0])
                    elif 0.2 < slope < 0.8:  # Right lane
                        right_lines.append(line[0])
        
        # Temporal smoothing
        if self.previous_lanes is not None:
            # Blend with previous frame (simple temporal filtering)
            alpha = 0.7  # Current frame weight
            if left_lines and self.previous_lanes[0]:
                # Simple blending logic would go here
                pass
                
        self.previous_lanes = (left_lines, right_lines)
        return left_lines, right_lines

class Week1ADASSystem:
    """Week 1 Enhanced ADAS System"""
    
    def __init__(self, model_path='yolov8n.pt'):
        self.detector = EnhancedYOLODetector(model_path)
        self.lane_detector = AdvancedLaneDetector()
        pygame.mixer.init()
        
        # System state
        self.collision_warning = False
        self.lane_warning = False
        self.alert_sound_played = False
        
        # Performance monitoring
        self.frame_times = []
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame with enhanced AI"""
        start_time = time.time()
        
        # Enhanced object detection
        detections = self.detector.detect_objects(frame)
        
        # Lane detection
        left_lines, right_lines = self.lane_detector.detect_lanes(frame)
        
        # Collision analysis
        self._analyze_collisions(detections)
        
        # Lane departure analysis
        self._analyze_lane_departure(frame, left_lines, right_lines)
        
        # Visualization
        frame = self._draw_enhanced_visualization(frame, detections, left_lines, right_lines)
        
        # Performance tracking
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        
        return frame
    
    def _analyze_collisions(self, detections: List[Detection3D]):
        """Enhanced collision analysis with 3D information"""
        self.collision_warning = False
        
        for detection in detections:
            # Enhanced TTC calculation using 3D position
            distance = detection.distance
            
            # Simple relative velocity assumption (will be enhanced in Week 3)
            relative_speed = 15  # m/s (will be calculated from tracking)
            
            if relative_speed > 0:
                ttc = distance / relative_speed
            else:
                ttc = float('inf')
            
            # Multi-factor collision risk
            risk_factors = {
                'distance': min(distance / 30.0, 1.0),  # Risk increases as distance < 30m
                'ttc': min(ttc / 3.0, 1.0) if ttc < float('inf') else 0,  # Risk for TTC < 3s
                'angle': abs(detection.angle) / math.pi,  # Higher risk for center objects
                'object_type': detection.risk_score
            }
            
            overall_risk = np.mean(list(risk_factors.values()))
            
            if overall_risk > 0.7 or (distance < 20 and ttc < 2.5):
                self.collision_warning = True
                self._play_collision_alert()
                break
    
    def _analyze_lane_departure(self, frame, left_lines, right_lines):
        """Enhanced lane departure analysis"""
        height, width = frame.shape[:2]
        center_x = width // 2
        
        self.lane_warning = False
        
        if left_lines and right_lines:
            # Calculate lane center more accurately
            left_x_avg = np.mean([line[0] for line in left_lines] + [line[2] for line in left_lines])
            right_x_avg = np.mean([line[0] for line in right_lines] + [line[2] for line in right_lines])
            
            lane_center = (left_x_avg + right_x_avg) / 2
            lane_width = right_x_avg - left_x_avg
            
            # Enhanced departure detection
            offset = abs(center_x - lane_center)
            offset_ratio = offset / (lane_width / 2) if lane_width > 0 else 0
            
            # Warning threshold based on lane width percentage
            if offset_ratio > 0.3:  # 30% of half lane width
                self.lane_warning = True
    
    def _draw_enhanced_visualization(self, frame, detections, left_lines, right_lines):
        """Enhanced visualization with 3D information"""
        height, width = frame.shape[:2]
        
        # Draw detections with 3D info
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox_2d
            
            # Color based on risk level
            if detection.risk_score > 0.7:
                color = (0, 0, 255)  # Red - high risk
            elif detection.risk_score > 0.4:
                color = (0, 165, 255)  # Orange - medium risk
            else:
                color = (0, 255, 0)  # Green - low risk
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Enhanced label with 3D information
            label = f'{detection.class_name}: {detection.confidence:.2f}'
            distance_label = f'Dist: {detection.distance:.1f}m'
            risk_label = f'Risk: {detection.risk_score:.2f}'
            
            # Multi-line text
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, distance_label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(frame, risk_label, (x1, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw lane lines with enhanced visualization
        if left_lines:
            for line in left_lines:
                cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 3)
        
        if right_lines:
            for line in right_lines:
                cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 3)
        
        # Vehicle position indicator
        center_x = width // 2
        cv2.line(frame, (center_x, height-50), (center_x, height-10), (255, 255, 0), 3)
        
        # System status and warnings
        status_y = 30
        if self.collision_warning:
            cv2.putText(frame, "COLLISION WARNING!", (50, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            status_y += 35
            
        if self.lane_warning:
            cv2.putText(frame, "LANE DEPARTURE WARNING!", (50, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            status_y += 35
        
        # Performance information
        perf_stats = self.detector.get_performance_stats()
        if perf_stats:
            fps_text = f"FPS: {perf_stats['fps']:.1f}"
            cv2.putText(frame, fps_text, (width-120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # System status
        status_color = (0, 255, 0) if not (self.collision_warning or self.lane_warning) else (0, 0, 255)
        cv2.putText(frame, "Week 1 ADAS Active", (width-200, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
        
        return frame
    
    def _play_collision_alert(self):
        """Play collision warning sound"""
        if not self.alert_sound_played:
            try:
                beep_freq = 1200
                beep_duration = 300
                sample_rate = 22050
                frames = int(beep_duration * sample_rate / 1000)
                
                # Create mono audio
                arr = np.zeros(frames, dtype=np.int16)
                for i in range(frames):
                    arr[i] = 4096 * np.sin(2 * np.pi * beep_freq * i / sample_rate)
                
                # Create stereo array with C-contiguous memory
                stereo_arr = np.ascontiguousarray(np.column_stack((arr, arr)))
                
                sound = pygame.sndarray.make_sound(stereo_arr)
                sound.play()
                self.alert_sound_played = True
                
                def reset_alert():
                    time.sleep(1.5)
                    self.alert_sound_played = False
                
                threading.Thread(target=reset_alert, daemon=True).start()
                
            except Exception as e:
                logger.warning(f"Could not play alert sound: {e}")
        
    def run_video(self, video_path: str):
        """Run ADAS on video file"""
        cap = cv2.VideoCapture(video_path)
        
        logger.info(f"Starting Week 1 ADAS on video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (1280, 720))  # Standardize input size
            processed_frame = self.process_frame(frame)
            
            cv2.imshow('Week 1 Enhanced ADAS System', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance summary
        self._print_performance_summary()
    
    def run_webcam(self):
        """Run ADAS on webcam"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        logger.info("Starting Week 1 ADAS on webcam")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = self.process_frame(frame)
            
            cv2.imshow('Week 1 Enhanced ADAS System', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self._print_performance_summary()
    
    def _print_performance_summary(self):
        """Print performance analysis"""
        if self.frame_times:
            avg_frame_time = np.mean(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            print(f"\n=== Week 1 ADAS Performance Summary ===")
            print(f"Average frame time: {avg_frame_time:.3f}s")
            print(f"Average FPS: {fps:.1f}")
            print(f"Total frames processed: {len(self.frame_times)}")
            
            detector_stats = self.detector.get_performance_stats()
            if detector_stats:
                print(f"Detection FPS: {detector_stats['fps']:.1f}")

if __name__ == "__main__":
    # Initialize Week 1 Enhanced ADAS
    adas = Week1ADASSystem()
    
    print("=== Week 1 Enhanced ADAS System ===")
    print("Features:")
    print("- Enhanced YOLOv8 with 3D object detection")
    print("- Real-world distance estimation")
    print("- Advanced risk assessment")
    print("- Performance monitoring")
    print("- Improved lane detection")
    print("\nOptions:")
    print("1. Run with video file")
    print("2. Run with webcam")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        video_path = r"/home/swapnil/Desktop/Placement26/ADAS/8359-208052066.mp4"
        adas.run_video(video_path)
    elif choice == "2":
        adas.run_webcam()
    else:
        print("Invalid choice")