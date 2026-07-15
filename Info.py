
System Context & Objective:
Act as a Principal Computer Vision and AI Automation Architect. You are being provided the architectural context for an enterprise-ready, modular industrial telemetry application. The goal is to build an automated computer vision engine that tracks analog gauge needles, parses their structural orientations into dynamic engineering metrics, avoids duplicate human calibration through self-managed JSON configurations, and provides an API/agent-ready runtime workflow for robotic integration.

The target environment contains multi-folder datasets where industrial monitoring cameras are rigidly mounted and physically static. Each camera watches exactly one distinct gauge dial face, capturing continuous sequential frames.

System Architecture and File Tree Layout:
The implementation must adhere strictly to the following directory structure:
axis-multi-gauge-reader/
├── gauges_data/             # Parent repository containing discrete gauge folders
│   ├── gauge1/
│   │   ├── config.json      # Dynamic target configuration automatically managed by the pipeline
│   │   ├── img1.jpg         # Representative baseline image used exclusively for anchor initialization
│   │   ├── img2.jpg         # Automatic downstream tracking frame asset
│   │   └── ...
│   └── gauge2/
│       ├── config.json
│       ├── frame_001.png
│       └── ...
└── src/
    ├── __init__.py
    ├── file_manager.py      # Abstracted I/O handler managing JSON configuration states and path resolution
    ├── calibration.py       # Geometric anchor assignment module providing pop-up dialogue inputs
    ├── engine.py            # High-performance CV image transformation and filtering array
└── main.py                  # Core automation loop and cluster manager orchestrating folder sweeps

Functional Specifications per Component Module:

1. src/file_manager.py
- Extract valid images sequentially based on a standard system sorting pattern.
- Look up `config.json` locally inside a specific sub-folder directory map.
- Safely parse dictionary assets from disk or generate a structural clean configuration file upon verification failure.

2. src/calibration.py
- Handle low-level graphic interactions over an active visual frame array utilizing OpenCV mouse callback bounds (`cv2.setMouseCallback`).
- Capture exactly 3 anchor coordinates sequentially: 1. Center Hub, 2. Min Scale Boundary, 3. Max Scale Boundary.
- Render descriptive text strings and unique color dots on target frames for instant verification.
- Right after the third click is validated, trigger an encapsulated Tkinter standalone dialog context (`tkinter.simpledialog.askfloat`) to securely poll the human operator for raw functional data constraints: "Scale Minimum Boundary Value" and "Scale Maximum Boundary Value".
- Bundle these assets inside a global configuration data matrix dictionary structure.

3. src/engine.py
- Initialize an active computing engine bound directly to the target configuration dictionary.
- Compute directional vectors utilizing raw angle values via `np.arctan2` mappings.
- Isolate the tracking perimeter by establishing an active concentric masking zone (donut model), dynamically eliminating core mechanical shaft artifacts (inner 25%) and edge bezel matrix noise (outer 5%).
- Flatten circular dial coordinates to horizontal matrix spaces using a high-fidelity linear polar transform mapping pipeline (`cv2.warpPolar`).
- Programmatically resolve dial face styles ('light' vs 'dark') via historical median gray value sweeps.
- Isolate the continuous linear needle bar from standard tick segments and alpha-numeric characters using a high-contrast `cv2.adaptiveThreshold` map followed by structural horizontal structural line element filters (`cv2.morphologyEx` using `cv2.MORPH_OPEN`).
- Perform mathematical angular calculations across valid coordinate boundaries using row-level pixel density checks.
- Map the resulting normalized angle directly onto the target operational metric limits via linear translation.

4. main.py
- Orchestrate sub-folder discovery sweeps within the parent `gauges_data/` folder directory.
- For each directory path: Scan for configurations. If nonexistent, spin up the interactive calibration pipeline using the first discoverable frame profile, auto-save the telemetry config dictionary to disk, and then instantly pass control forward.
- Initialize the computing workspace using the newly created or pre-cached configuration parameters.
- Loop smoothly through all available pictures in the folder without breaking execution loops or triggering annoying repetitive user dialog boxes.
- Project tracking line overlays and active digital readouts onto dynamic frames, printing telemetry results out cleanly to standard logger outputs.

This design matrix is built for high automated flexibility. AI Agents must be able to inject config matrices or parse variables directly out of storage blocks cleanly without UI interruption when files are pre-cached.

Acknowledge that you completely understand this structural architecture blueprint, the interaction between files, the conditional JSON checking logic, and the target automated Agent execution pattern.








1. src/file_manager.py
import os
import json
import logging

logger = logging.getLogger("FileManagerModule")

def get_gauge_images(folder_path):
    """Scans the folder and returns a list of valid image paths."""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    # Sort files to ensure consistency (e.g., lowest name index first for calibration)
    return sorted(files)

def load_gauge_config(folder_path):
    """
    Checks if config.json exists in the gauge folder.
    Returns the parsed dictionary if found, otherwise returns None.
    """
    config_path = os.path.join(folder_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            logger.info(f"Existing configuration loaded successfully from: {config_path}")
            return config_data
        except Exception as e:
            logger.error(f"Failed to read existing config file: {e}")
            return None
    return None

def save_gauge_config(folder_path, config_data):
    """Saves the calibration coordinates and limits into a local folder config.json."""
    config_path = os.path.join(folder_path, "config.json")
    try:
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        logger.info(f"New configuration auto-saved to disk at: {config_path}")
    except Exception as e:
        logger.error(f"Failed to write config file to disk: {e}")











2. src/calibration.py
import cv2
import logging
import tkinter as tk
from tkinter import simpledialog

logger = logging.getLogger("CalibrationModule")

class CalibrationGUI:
    def __init__(self, window_name, image):
        self.window_name = window_name
        self.image = image.copy()
        self.points = []
        self.labels = ["CENTER HUB", "MIN VALUE (Zero/E)", "MAX VALUE (Full/F)"]

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 3:
            self.points.append((x, y))
            idx = len(self.points) - 1
            colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
            
            cv2.circle(self.image, (x, y), 5, colors[idx], -1)
            cv2.putText(self.image, f" {self.labels[idx]}", (x + 10, y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
            cv2.imshow(self.window_name, self.image)
            logger.info(f"Registered {self.labels[idx]} at: ({x}, {y})")

def run_interactive_calibration(frame, folder_name):
    """
    Launches interactive calibration UI for a single representative image.
    Prompts user for engineering unit boundaries.
    """
    window_name = f"Calibrating [{folder_name}] - Click Center -> Min -> Max. 'q' to abort."
    gui = CalibrationGUI(window_name, frame)
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, gui.mouse_callback)
    cv2.imshow(window_name, gui.image)
    
    while len(gui.points) < 3:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyWindow(window_name)
    
    if len(gui.points) < 3:
        raise ValueError(f"User aborted calibration sequence for folder: {folder_name}")
        
    root = tk.Tk()
    root.withdraw() 
    
    scale_min = simpledialog.askfloat("Scale Config", f"[{folder_name}] Enter numerical value at MIN point:", initialvalue=0.0)
    scale_max = simpledialog.askfloat("Scale Config", f"[{folder_name}] Enter numerical value at MAX point:", initialvalue=100.0)
    
    root.destroy()
    
    if scale_min is None: scale_min = 0.0
    if scale_max is None: scale_max = 100.0
    
    # Pack parameters inside an Agent-friendly structured dictionary format
    config_data = {
        "center_point": gui.points[0],
        "min_point": gui.points[1],
        "max_point": gui.points[2],
        "scale_min": scale_min,
        "scale_max": scale_max,
        "clockwise": True,
        "round_to_decimals": 2
    }
    
    return config_data
















3. src/engine.py
import cv2
import numpy as np

class IndustrialGaugeEngine:
    """Core CV engine operating directly on dictionary payload configs for Agent flexibility."""
    
    def __init__(self, config_dict):
        self.center = (float(config_dict["center_point"][0]), float(config_dict["center_point"][1]))
        self.min_point = (float(config_dict["min_point"][0]), float(config_dict["min_point"][1]))
        self.max_point = (float(config_dict["max_point"][0]), float(config_dict["max_point"][1]))
        self.clockwise = config_dict.get("clockwise", True)
        self.scale_min = config_dict.get("scale_min", 0.0)
        self.scale_max = config_dict.get("scale_max", 100.0)
        self.round_to_decimals = config_dict.get("round_to_decimals", 2)

        self.angle_min = self._get_angle(self.min_point)
        self.angle_max = self._get_angle(self.max_point)

        dist_min = np.linalg.norm(np.array(self.min_point) - np.array(self.center))
        dist_max = np.linalg.norm(np.array(self.max_point) - np.array(self.center))
        self.max_radius = int(max(dist_min, dist_max) * 0.95)
        self.min_radius = int(min(dist_min, dist_max) * 0.25) 

    def _get_angle(self, point):
        dy = point[1] - self.center[1]
        dx = point[0] - self.center[0]
        return np.degrees(np.arctan2(dy, dx)) % 360

    def calculate_reading(self, frame):
        """Processes static or dynamic frames against a pre-calibrated geometric configuration matrix."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        polar_res = cv2.warpPolar(gray, dsize=(self.max_radius, 360), 
                                  center=self.center, 
                                  maxRadius=self.max_radius, 
                                  flags=cv2.WARP_POLAR_LINEAR + cv2.INTER_CUBIC)

        cropped_polar = polar_res[:, self.min_radius:self.max_radius]

        sample_area = cropped_polar[:, int(cropped_polar.shape[1]*0.2):int(cropped_polar.shape[1]*0.8)]
        median_brightness = np.median(sample_area)
        dial_type = 'light' if median_brightness > 120 else 'dark'

        if dial_type == 'light':
            binary_input = cv2.bitwise_not(cropped_polar)
        else:
            binary_input = cropped_polar.copy()

        binary = cv2.adaptiveThreshold(
            binary_input, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, -5
        )

        active_width = cropped_polar.shape[1]
        kernel_len = max(5, int(active_width * 0.35)) 
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        filtered_polar = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

        angular_profile = np.sum(filtered_polar, axis=1)

        best_angle = -1
        max_score = -1

        for degree in range(360):
            if self.clockwise:
                total_arc = (self.angle_max - self.angle_min) % 360
                current_arc = (degree - self.angle_min) % 360
            else:
                total_arc = (self.angle_min - self.angle_max) % 360
                current_arc = (self.angle_min - degree) % 360

            if current_arc <= total_arc:
                score = angular_profile[degree]
                if score > max_score:
                    max_score = score
                    best_angle = degree

        if best_angle == -1 or max_score == 0:
            return -1.0, 0.0, dial_type

        if self.clockwise:
            travel = (best_angle - self.angle_min) % 360
        else:
            travel = (self.angle_min - best_angle) % 360

        percentage = travel / total_arc
        actual_value = self.scale_min + (percentage * (self.scale_max - self.scale_min))
        actual_value = max(self.scale_min, min(self.scale_max, actual_value))
        
        return round(actual_value, self.round_to_decimals), float(best_angle), dial_type















4. main.py
import os
import cv2
import numpy as np
import logging

from src.file_manager import get_gauge_images, load_gauge_config, save_gauge_config
from src.calibration import run_interactive_calibration
from src.engine import IndustrialGaugeEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MultiFolderOrchestrator")

def process_single_image(image_path, engine):
    """Loads an image, processes it with the engine, and renders an overlay output window."""
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error(f"Failed to read raw matrix data from: {image_path}")
        return None

    value, angle_deg, dial_mode = engine.calculate_reading(frame)
    
    if value >= 0:
        # Re-project vector calculation for drawing needle line overlay
        rad = np.radians(angle_deg)
        target_length = engine.max_radius * 0.95
        tx = int(engine.center[0] + target_length * np.cos(rad))
        ty = int(engine.center[1] + target_length * np.sin(rad))
        center_int = (int(engine.center[0]), int(engine.center[1]))

        cv2.line(frame, center_int, (tx, ty), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (tx, ty), 6, (0, 0, 255), -1)
        
        display_text = f"VALUE: {value} | DIAL: {dial_mode.upper()}"
        cv2.putText(frame, display_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        logger.info(f"Image processed: {os.path.basename(image_path)} -> Result: {value}")
    else:
        cv2.putText(frame, "NEEDLE LOSS DETECTED", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        logger.warning(f"Needle lost in scene frame: {image_path}")

    return frame

def main():
    # Base directory holding all targeted subfolders
    root_data_dir = "gauges_data"
    
    if not os.path.exists(root_data_dir):
        logger.error(f"Root data directory '{root_data_dir}' not found. Please create it.")
        return

    # Scan root directory for inner subdirectories
    subfolders = [os.path.join(root_data_dir, d) for d in os.listdir(root_data_dir) if os.path.isdir(os.path.join(root_data_dir, d))]
    logger.info(f"Discovered {len(subfolders)} gauge folders inside repository data paths.")

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        logger.info(f"-------------------------------------------------------")
        logger.info(f"Processing Subfolder Array Target: [{folder_name}]")
        
        # Get all pictures grouped inside this specific subdirectory
        images = get_gauge_images(folder)
        if not images:
            logger.warning(f"No image files discovered inside [{folder_name}]. Skipping.")
            continue

        # Step 1: Detect or Create Local JSON Config Payload
        config_data = load_gauge_config(folder)
        
        if config_data is None:
            logger.info(f"Config mapping missing for [{folder_name}]. Initializing manual GUI setup window...")
            # Use the first historical image as the rigid base anchor for calibration clicks
            calibration_base_img = cv2.imread(images[0])
            if calibration_base_img is None:
                logger.error(f"Failed to load first baseline image {images[0]} for calibration. Skipping folder.")
                continue
            
            try:
                # Run the UI setup sequence
                config_data = run_interactive_calibration(calibration_base_img, folder_name)
                # Save configuration maps to storage automatically
                save_gauge_config(folder, config_data)
            except Exception as e:
                logger.error(f"Skipping folder setup execution loop due to tracking error: {e}")
                continue

        # Step 2: Initialize CV Engine with loaded or created configuration maps
        engine = IndustrialGaugeEngine(config_data)

        # Step 3: Run entirely automated computation pipelines over remaining files
        logger.info(f"Starting automated data processing routine for remaining images in [{folder_name}]...")
        for img_path in images:
            output_frame = process_single_image(img_path, engine)
            
            if output_frame is not None:
                window_title = f"Telemetry Analytics Feed: {folder_name}"
                cv2.imshow(window_title, output_frame)
                
                # Show frame for a short delay (e.g., 800ms) to allow inspection, press 'q' to break to next folder
                if cv2.waitKey(800) & 0xFF == ord('q'):
                    break
        
        cv2.destroyAllWindows()
        logger.info(f"Completed batches inside target container module: [{folder_name}]")

    logger.info("All target dataset profiles resolved successfully.")

if __name__ == "__main__":
    main()
