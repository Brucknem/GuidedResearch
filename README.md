# Automated Camera Stabilization and Calibration for Intelligent Transportation Systems
I have successfully finished my Guided Research within the [PROVIDENTIA](https://www.bmvi.de/SharedDocs/DE/Artikel/DG/AVF-projekte/providentia-plusplus.html) project during the winter term of 2020/2021.  
The report is included in this repository, and the implementation is linked.

***

## Background
Within the [PROVIDENTIA](https://www.bmvi.de/SharedDocs/DE/Artikel/DG/AVF-projekte/providentia-plusplus.html) project, a section of the highway A9 between Munich and Nuremberg was converted to a testing site for autonomous driving.
As part of this, a large sensor network system has been set up along the highway to allow monitoring and steering of traffic as well as to improve the coordination between autonomous and traditional cars.
The primary task of the intelligent system is to create a digital traffic twin that accurately represents the physical road situation in real-time.
Based on this digital twin, the smart infrastructure can provide a far-reaching and comprehensive view to the drivers and autonomous cars in order to improve their situational awareness within the current traffic environment.
A video about the PROVIDENTIA project is available on https://youtu.be/4oCnQlGFuc4.

## Description
A key challenge lies in the reliable and accurate calibration of the different sensors.
The calibration is especially challenging when the sensor is subject to real-life disturbances like vibration of its mounting pole caused by wind or displacements due to temperature expansion.
The aim of this Master’s thesis is to investigate the feasibility to automatically stabilize and calibrate a shaking camera using an additional IMU sensor that delivers measurements on the disturbances.

## Tasks
- Familiarization with stabilization and calibration methods via literature research
- Development of an experimental setup consisting of an oscillating platform equipped with a camera and an IMU
- Development of an approach to automatically stabilize and calibrate the camera
- Evaluation of the concept using real-life data

***

## Implementation
- [Dynamic Stabilization](https://github.com/Brucknem/DynamicStabilization)
- [Static Calibration](https://github.com/Brucknem/DynamicStabilization)

***

## [Literature](https://github.com/Brucknem/StaticCalibration)
