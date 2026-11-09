# Orion's Shield

A real-time surveillance web app. It recognises faces from uploaded images, detects weapons and threat objects in connected camera feeds, logs each event with a screenshot, and sends SMS alerts.

## Overview

Authenticated users log in to a Django web app that combines:

- **Face recognition** with InsightFace (buffalo_s model) for faces uploaded to the system.
- **Weapon and threat detection** with a custom-trained YOLOv5n model (FP32 PyTorch weights) for live camera feeds.
- **Event logging** with timestamps and screenshots. Faces are recency-checked over 30 minutes, and weapons over 2 minutes.
- **SMS alerts** through Twilio about the most recent log entry.

The weapon model was trained on 14 classes: knife, knife_in_hand, pistol, pistol_in_hand, rifle, rifle_in_hand, sniper, shotgun, shotgun_in_hand, RPG, MG, Mounted_MG, Drone, Front_Face, and Side_Face. See [Results](#results) for which classes have usable training data.

Architecture diagrams:
- [Detection and recognition pipeline](https://github.com/Daneil98/FORS/blob/master/Facial%20recognition%20and%20weapon%20detection%20algorithmic%20pipeline.png)
- [Django system architecture](https://github.com/Daneil98/FORS/blob/master/Django%20System%20Architecture.png)
- [Feature screenshots](https://github.com/Daneil98/FORS/tree/master/Orion's%20Shield)

## Features

- **User authentication:** sign-up, login, and logout with hashed password storage.
- **Face recognition:** matches faces from uploaded images against people seen on camera.
- **Weapon detection:** detects threat objects in live camera feeds.
- **Notification logs:** stores recognised people and detected weapons with timestamps and frames.
- **SMS notifications:** sends an alert about the latest log entry to a preconfigured number.

## Requirements

- Python 3.9
- CMake and the C++ build tools (Visual Studio C++ SDK on Windows)
- A CUDA-capable GPU is optional. In our tests, recognition ran faster on CPU than on GPU (see Results).
- A Twilio account, if you want SMS alerts

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Daneil98/FORS
   cd FORS
   ```

2. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Clone YOLOv5 into the project directory and install its dependencies:

   ```bash
   git clone https://github.com/ultralytics/yolov5
   cd yolov5
   pip install -r requirements.txt
   cd ..
   ```

4. Place the trained weights file, `best.pt`, in the project root (the same folder as `manage.py`).

5. Create a `.env` file in the project root with the variables below. Do not commit this file.

   ```
   URL=                 # camera stream URL, for example over Wi-Fi
   account_sid=         # Twilio account SID
   auth_token=          # Twilio auth token
   sending_number=      # Twilio sending number
   receiving_number=    # number that receives alerts
   ```

6. Apply database migrations:

   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

7. Create an admin account. This is the easiest way to log in and manage users:

   ```bash
   python manage.py createsuperuser
   ```

8. Start the server:

   ```bash
   python manage.py runserver
   ```

   Open http://127.0.0.1:8000 and log in with the account you created.

## Results

All measurements were taken on an Intel i5-8250U (4 threads, no GPU for the detector), 8 GB RAM, NVIDIA MX150 (2 GB), Windows 11, PyTorch 2.3.0, and ONNX Runtime 1.19.2. Absolute latencies varied between sessions because of CPU thermal state, so the speed-up ratios are the more reliable figures.

### Detection accuracy

Measured on the 928-image validation split (confidence 0.001, NMS IoU 0.60).

| Configuration | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| PyTorch FP32, 640 (shipped weights) | 0.6675 | 0.4323 | 0.675 | 0.665 |
| ONNX FP32, 640 | 0.6667 | 0.4245 | 0.683 | 0.676 |
| ONNX FP32, 320 | 0.6047 | 0.3864 | 0.689 | 0.607 |
| INT8, 640, reduce_range=True | 0.6576 | 0.4168 | 0.696 | 0.655 |
| INT8, 320, reduce_range=True | 0.5990 | 0.3843 | 0.693 | 0.593 |

Quantising the whole graph, including the detection head, dropped mAP@0.5 to 0.0000.

### Speed

| Change | Latency | Speed-up |
|---|---|---|
| PyTorch to ONNX Runtime, 640 | 104.1 → 58.5 ms | 1.78× |
| FP32 to INT8 (reduce_range=True), 640 | 60.3 → 48.6 ms | 1.24× |
| FP32 to INT8, 320 | 15.5 → 14.1 ms | 1.10× |
| FP32, 640 to FP32, 320 | 60.3 → 18.3 ms | 3.29× |

Lowering the input resolution gave the largest speed-up, but it also cost the most accuracy (about 0.06 mAP@0.5). The ONNX Runtime change cost almost no accuracy.

Model size: FP32 is 7.59 MB, and INT8 is 2.56 MB.

### Quantisation

- The shipped `quantize_dynamic` call targets `nn.Linear` layers, and this model has none. The shipped model is therefore identical to FP32, and the weights in this repository are not quantised.
- The INT8 results above come from a separate static quantisation experiment. It keeps 57 of 60 convolutions quantised.
- INT8 accuracy loss is uneven across classes: shotgun dropped by 0.465 AP, knife by 0.329, and Drone by 0.031.
- Percentile and entropy calibration failed on this hardware, so MinMax calibration was used.

### Concurrency and latency

- Detection alone ran at 75.6 FPS. With recognition running concurrently on the GPU, it ran at 67.9 FPS.
- The combined steady-state pipeline ran at 49.7 FPS, with recognition running on 1 frame in 20.
- Recognition latency (median, 91 crops):

  | Configuration | Median latency |
  |---|---|
  | GPU, all 5 sub-models (as deployed) | 309.0 ms |
  | GPU, detection and recognition only | 74.0 ms |
  | CPU, detection and recognition only | 25.2 ms |

  The GPU path was 2.9× slower than CPU for this model, and unused sub-models added 4.2× to the GPU time.

### Alerts

- Capture to alert took 222 ms on average over 40 events. SMS dispatch accounted for 191 ms of that.
- Sending the SMS in the main frame loop cut throughput from 96 FPS to 4.4 FPS during each alert.
- Sending the SMS from a background thread kept throughput at 61.5 FPS and cut frame-loop stalls by 93%. On a real network, throughput during an alert rose from 7.3 to 56.6 FPS.

### Face recognition

- The gallery has 2 identities and no impostor set, so this project does not report a recognition accuracy figure.
- The cited component benchmarks are ArcFace (99.83% on LFW) and MobileFaceNet (99.55% on LFW). These are published results for those models, not measured on this system.
- The Front_Face gate has precision 0.19 and recall 0.87. About 4 in 5 crops sent to recognition are not faces.

### Detection failure analysis

- Drone and Front_Face are confused with each other in both directions (7% and 9%).
- Side_Face fires on background 24% of the time. It has zero training instances, so this is a known failure, not a model result.
- Handguns had the highest missed-detection rates: pistol_in_hand at 0.44 and pistol at 0.38.

### Dataset

6,346 images (5,418 train, 928 validation, 0 test) with 7,313 instances. Fourteen of the 15 classes are annotated. Front_Face has 71 training instances.

## Known limitations

- Side_Face has no training data.
- There is no held-out test split. The validation split was used for all reported numbers.
- The face gallery is too small to measure recognition accuracy.
- The weights are FP32 and are not quantised. The INT8 and ONNX results are experiments, not the shipped configuration.
- Recognition is currently slower on GPU than on CPU.
