# AI-BASED-POTHOLE-DETECTION-USING-DRONE
Project Overview

This project presents a **real-time pothole detection system** using **Deep Learning and Computer Vision**.
The system captures live video from a **drone/mobile camera**, streams it through **OBS Virtual Camera**, and processes the video using a **YOLO-based object detection model**.
Detected potholes are highlighted with bounding boxes, classified based on severity, and logged into a CSV file for reporting and analysis. The results are displayed through an interactive **Streamlit dashboard**.

 Objectives

* Detect potholes in real time using deep learning
* Capture live video feed using drone/mobile camera
* Classify potholes based on severity level
* Display detection results on a dashboard
* Log detection details into a CSV file
* Improve road monitoring efficiency

---
Technologies Used

* **Python**
* **OpenCV**
* **YOLO (Ultralytics)**
* **Streamlit**
* **OBS Studio**
* **scrcpy**
* **Roboflow (Dataset Creation & Annotation)**
* **NumPy**
* **Pandas**

---

System Architecture

Drone / Mobile Camera
⬇
scrcpy (Screen Mirroring)
⬇
OBS Virtual Camera
⬇
Python (OpenCV + YOLO Model)
⬇
Streamlit Dashboard
⬇
CSV Logging System

---
Project Features

✔ Real-time pothole detection
✔ Severity classification (Low, Medium, High)
✔ Live video streaming via OBS
✔ Interactive Streamlit dashboard
✔ FPS monitoring
✔ CSV logging of detections
✔ Confidence score display
✔ Bounding box visualization

---
 Severity Classification

Potholes are classified based on bounding box size:

* **Low Severity**
* **Medium Severity**
* **High Severity**

This helps prioritize road maintenance.

---
GPS Integration (Optional / Future Enhancement)

The system supports geo-tagging of detected potholes using GPS coordinates.
Coordinates can be logged along with detection data for location-based reporting.

---
 Installation

Install required libraries:

```bash
pip install streamlit opencv-python ultralytics pandas numpy
```

Install **OBS Studio** and **scrcpy**.

---
How to Run the Project

1. Connect mobile/drone camera
2. Mirror screen using **scrcpy**
3. Open **OBS Studio**
4. Add **Window Capture (scrcpy window)**
5. Click **Start Virtual Camera**
6. Run the Streamlit app:

streamlit run app.py

7. Select camera index from sidebar
8. Start detection

---

Dataset Information

* Dataset created using **Roboflow**
* Images manually labeled
* Multiple training iterations performed
* Model optimized for better accuracy

---

Model Performance

* **Model:** YOLO (Ultralytics)
* **Accuracy:** ~87%
* **Real-Time Processing:** Supported
* **Confidence Threshold:** Adjustable

---
 Future Enhancements

* GPS-based geo-tagging
* Heatmap visualization
* Automatic pothole counting
* Snapshot saving
* Integration with municipal reporting systems
* Mobile application support

---
Team Members & Roles

**1. Model Training & System Coordination**

* Dataset training and testing
* Video streaming integration
* Overall system coordination

**2. Hardware & Drone Integration**

* Drone research
* Connectivity setup
* Hardware testing

**3. UI & System Integration**

* Streamlit dashboard design
* System connectivity setup

**4. Dataset & Documentation**

* Dataset creation and labeling
* Model validation
* Documentation support

---
Applications

* Smart City Infrastructure
* Road Maintenance Monitoring
* Traffic Safety Systems
* Municipal Road Inspection
* Infrastructure Management

---
 Project Highlights

* Real-time detection system
* Drone-assisted monitoring
* AI-powered road inspection
* Cost-effective solution
* Scalable architecture

---
 License

This project is developed for academic and research purposes.
 

