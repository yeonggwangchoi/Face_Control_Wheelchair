# Face Control Wheelchair

A computer vision-based wheelchair control system that enables hands-free operation through facial movements and head gestures. The system uses facial landmark detection to interpret user intent and includes safety features like traffic light detection and obstacle avoidance.

## Features

- **Face-based Control**: Control wheelchair movement using head position and facial gestures
- **Traffic Light Detection**: Automatically stops when red traffic lights are detected
- **Obstacle Avoidance**: Detects obstacles in the forward path and prevents collisions
- **Dual-System Architecture**: Distributed processing between Main PC and Raspberry Pi for optimal performance
- **Real-time Processing**: 60 FPS camera capture with low-latency response

## System Architecture

```
┌─────────────────┐     Serial      ┌─────────────────┐     Serial      ┌─────────────────┐
│   Raspberry Pi  │ ──────────────► │     Main PC     │ ──────────────► │   ATmega128     │
│    (Sub PC)     │   Direction     │                 │   Motor Cmd     │ Motor Controller│
│                 │   Commands      │                 │                 │                 │
│  - Face Detect  │                 │  - Safety Check │                 │  - Motor Drive  │
│  - Direction    │                 │  - Traffic Light│                 │                 │
│    Calculation  │                 │  - Obstacle Det │                 │                 │
└─────────────────┘                 └─────────────────┘                 └─────────────────┘
        │                                   │
        ▼                                   ▼
   ┌─────────┐                        ┌──────────┐
   │ Camera 1│                        │ Camera 2 │
   │ (Face)  │                        │ (Road)   │
   └─────────┘                        └──────────┘
```

### Component Roles

| Component | Role | Key Functions |
|-----------|------|---------------|
| **Sub PC (Raspberry Pi)** | Face detection & direction calculation | Captures face, extracts 68 facial landmarks, determines movement direction |
| **Main PC** | Safety system & ML inference | Traffic light detection, obstacle detection, safety override |
| **ATmega128** | Motor control | Receives final commands, drives wheelchair motors |

## Hardware Requirements

- **Main PC**: Desktop/Laptop with GPU (recommended for YOLO inference)
- **Sub PC**: Raspberry Pi 4 or equivalent
- **Microcontroller**: ATmega128
- **Cameras**: 2x USB cameras (640x480 resolution)
- **Serial Connections**: USB-to-Serial adapters

### Serial Port Configuration

| Connection | Windows | Linux | Baud Rate |
|------------|---------|-------|-----------|
| Main PC ↔ Raspberry Pi | COM12 | /dev/ttyUSB0 | 9600 |
| Main PC ↔ ATmega128 | COM15 | /dev/ttyUSB1 | 9600 |
| Raspberry Pi Serial | - | /dev/ttyAMA0 | 9600 |

## Software Requirements

### Dependencies

```bash
pip install opencv-python dlib torch ultralytics pyserial numpy
```

### ML Models

The following pre-trained models are required:

| Model | Size | Purpose | Location |
|-------|------|---------|----------|
| `mmod_human_face_detector.dat` | 729 KB | dlib CNN face detection | `Face_Control_Wheelchair_Main_PC/` |
| `shape_predictor_68_face_landmarks.dat` | 99.7 MB | 68 facial landmark prediction | `Face_Control_Wheelchair_Main_PC/` |
| `yolov8n.pt` | 6.5 MB | Obstacle detection (COCO 80 classes) | `Yolo_model/` |
| Custom traffic light model | 6.2 MB | Traffic light detection (3 classes) | `Traffic_Light/` |

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yeonggwangchoi/Face_Control_Wheelchair.git
   cd Face_Control_Wheelchair
   ```

2. **Install dependencies**
   ```bash
   pip install opencv-python dlib torch ultralytics pyserial numpy
   ```

3. **Download dlib models** (if not included)
   - [mmod_human_face_detector.dat](http://dlib.net/files/mmod_human_face_detector.dat.bz2)
   - [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

4. **Configure serial ports**
   - Edit `atmegaserial.py` and `raspberryserial.py` to match your system's serial port names

## Usage

### Running the Main PC

```bash
cd Face_Control_Wheelchair_Main_PC
python main.py
```

### Running the Sub PC (Raspberry Pi)

```bash
cd Face_Control_Wheelchair_Sub_PC
python main.py
```

### Control Commands

| Command | Action | Trigger |
|---------|--------|---------|
| `G` | Go forward | Face centered |
| `S` | Stop | Emergency (mouth open) or safety override |
| `L` | Turn left | Head turned left |
| `R` | Turn right | Head turned right |

### Exit

Press `q` to quit the application.

## How It Works

### Face Direction Detection

The system uses 68 facial landmarks detected by dlib:

1. **Landmark Point 34** (nose tip): Used as the reference point for horizontal position
2. **Center Line (MID_X=160)**: Virtual center line of the camera frame
3. **Direction Logic**:
   - Nose left of center → Turn Left (`L`)
   - Nose right of center → Turn Right (`R`)
   - Nose centered → Go Forward (`G`)

### Emergency Stop

- **Mouth Open Check**: Points 63-67 (inner lips) - if vertical distance exceeds 15 pixels, triggers emergency stop
- **Useful for**: Intentional mouth-open gesture to stop immediately

### Safety Systems

1. **Traffic Light Detection**
   - Custom YOLOv8 model trained on traffic light dataset
   - Classes: Green, Red, Yellow
   - Action: If red light detected AND motor command is "Go" → Override to "Stop"

2. **Obstacle Detection**
   - YOLOv8 with forward detection polygon mask
   - Detects objects in the wheelchair's forward path
   - Action: If obstacle detected AND motor command is "Go" → Override to "Stop"

## Project Structure

```
Face_Control_Wheelchair/
├── Face_Control_Wheelchair_Main_PC/
│   ├── main.py              # Main control loop
│   ├── camera.py            # Camera & ML inference
│   ├── atmegaserial.py      # ATmega128 communication
│   ├── raspberryserial.py   # Raspberry Pi communication
│   └── *.dat                # dlib models
├── Face_Control_Wheelchair_Sub_PC/
│   ├── main.py              # Sub PC control loop
│   ├── camera.py            # Face detection module
│   ├── raspberryserial.py   # Serial communication
│   └── *.dat                # dlib models
├── Traffic_Light/
│   ├── *.pt                 # Traffic light YOLO models
│   ├── custom_yolov8.yaml   # Model architecture
│   └── data.yaml            # Training data config
├── Yolo_model/
│   ├── yolov8n.pt           # Nano model (default)
│   ├── yolov8s.pt           # Small model
│   └── yolov8l.pt           # Large model
└── README.md
```

## Model Training

To train the traffic light detection model:

```bash
cd Face_Control_Wheelchair_Main_PC
jupyter notebook custom_train.ipynb
```

The notebook contains the complete training pipeline using Ultralytics YOLOv8.

## Configuration

### Camera Resolution

| System | Camera 1 | Camera 2 |
|--------|----------|----------|
| Main PC | 640x480 @ 60fps | 640x480 @ 60fps |
| Sub PC | 320x240 | 640x480 |

### Detection Thresholds

- **Direction threshold**: ±10-20 pixels from center
- **Traffic light confidence**: Model default
- **Obstacle detection**: Forward polygon mask

## Troubleshooting

### Serial Port Issues
- Verify correct port names in `atmegaserial.py` and `raspberryserial.py`
- Check USB connections and permissions (`sudo chmod 666 /dev/ttyUSB*` on Linux)

### Camera Not Detected
- Verify camera indices in `camera.py` (`cv2.VideoCapture(0)`, `cv2.VideoCapture(1)`)
- Check camera permissions on Linux

### Model Loading Errors
- Ensure all `.dat` and `.pt` files are in the correct directories
- Check file paths in `camera.py` match your system

## License

This project is for educational and research purposes.

## Acknowledgments

- [dlib](http://dlib.net/) - Face detection and landmark prediction
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [OpenCV](https://opencv.org/) - Computer vision library
