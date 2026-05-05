# lane-keeping-assist-carmaker-simulink
Development of a camera-based Lane Keeping Assist (LKA) system using Python, MATLAB/Simulink and IPG CarMaker (CM4SL)
# 🚗 Lane Keeping Assist System (LKA)

This project presents the development of a camera-based Lane Keeping Assist (LKA) system using Python, MATLAB/Simulink, and IPG CarMaker.

## 🔍 Overview
The system detects lane markings from camera images and computes a steering angle to keep the vehicle centered within the lane.

## ⚙️ Workflow
1. Prototype development in Python (PyCharm)
2. Image processing:
   - Grayscale conversion
   - Gaussian filtering
   - Canny edge detection
   - ROI selection
   - Hough Transform
3. Lane geometry estimation:
   - Lateral offset
   - Heading angle
4. Controller design:
   - P controller
   - PI controller
5. Migration to MATLAB/Simulink
6. Integration with IPG CarMaker (co-simulation)

## 🧠 Key Features
- Camera-based lane detection
- Steering angle computation
- Signal filtering (Butterworth, Savitzky-Golay)
- Open-loop simulation in CarMaker

## 📊 Results
The PI controller reduces steady-state lateral error and improves lane tracking stability compared to a P controller.

## 🛠 Tools & Technologies
- Python (OpenCV)
- MATLAB / Simulink
- IPG CarMaker
- PyCharm

## 📌 Future Work
- Closed-loop control
- Deep learning-based lane detection (CNN)
- Sensor fusion (Camera + Radar/LiDAR)

---

## 📷 Demo
(Add images or simulation video here)
