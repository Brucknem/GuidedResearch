https://thesisguide.org/2014/10/13/thesis-architecture/

# Introduction
- [ ] Whats the problem statement_
  - [ ] ITS use cameras as one sensor type
  - [ ] Cameras are subject to real-life disturbances
  - [ ] Dynamic measurements are noisy
  - [ ] Cameras drift and static calibration becomes noisy
- [ ] Why is this a problem
  - [ ] Tracking and predicting becomes noisy
  - [ ] Camera to digital twin calibration becomes worse over time
  - [ ] Inter camera calibration becomes worse over time
- [ ] Contributions to solve these problems
  - [ ] Dynamic stabilization
    - [ ] Vision based, digital image stabilization approach
    - [ ] Feature matching and homographic transformation
  - [ ] Static calibration
    - [ ] HD map based approach
    - [ ] Optimization algorithm, reprojection error between map and video 
    - [ ] Landmark extraction, mapping, pose estimation

# Terms and Definitions
- [ ] Most likely omitted
- [ ] Explanations at relevant locations in approach

# Related Work
- [ ] Where differs?
  - [ ] Mostly at usage in ITS on highway for traffic control
  - [ ] Not huge movements as in most papers, but small scale disturbances
  - [ ] Mostly stationary cameras, but huge frustums
  - [ ] HD map for calibration
- [ ] Compare to feature detector studies in literature
- [ ] Literature research ongoing, I have to search through my list of references which was relevant for my work

# Approach
- [ ] Dynamic stabilization
  - [ ] Vision based, digital image stabilization approach
  - [ ] Feature detection
  - [ ] Feature matching
  - [ ] Homographic transformation
  - [ ] Multiple feature detectors  
- [ ] Static calibration
  - [ ] HD map based approach
  - [ ] Optimization algorithm, reprojection error between map and video 
  - [ ] Landmark extraction, mapping, pose estimation
  - [ ] Watersheder for pixel marking

# Evaluation
- [ ] Dynamic stabilization
  - [ ] Compare different feature detectors
  - [ ] Compare optical flow: Mean & std
  - [ ] Track features and calculate path smoothness
- [ ] Static calibration
  - [ ] Compare to GPS position
  - [ ] Compare to Google Maps/Open Street Map
  - [ ] Compare to Gyro Rotation
  - [ ] Compare by hand calibration
  - [ ] Calculate expectable error

# Future Work
- [ ] Dynamic stabilization
  - [ ] Dynamically pick best feature detector based on optical flow performance
  - [ ] Test on different weather/lighting conditions
  - [ ] IMU based sensor fusion (Only for paper, no real lookout)
  - [ ] Optical FLow / Warp field stabilization
  - [ ] ML CNN
    - [ ] Image MSE for calibration (unsupervised)
    - [ ] Logistic regression on pose (supervised)
- [ ] Static calibration
  - [ ] Object height and heading
  - [ ] Extend to global refinement of multiple cameras
  - [ ] After initial by hand calibration with watersheder: Re-detection in new keyframes:
    - [ ] Automatic recalibration
    - [ ] Per frame pose estimation
  - [ ] Automatic detection of more landmarks after initial calibration
  - [ ] Automatic mapping pixel <-> object
- [ ] Vehicle pose estimation
  - [ ] Use HD map roads and camera look-at vector
  - [ ] Calculate ground plane/description from map
  - [ ] Calculate depth
- [ ] Hough lines flucht punkte
- [ ] New HD map

# Conclusion
- [ ] Depends on results