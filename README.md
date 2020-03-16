# VR head tracking
Track head position and rotation form VR Oculus Quest.
In this project, we collect the position (X, Y, Z values) and rotation(pitch, yaw, roll) of the head and the controllers.

## Data Analysis
In the "dataAnalysis.py" file, there is a script used for data analysis. We follow these steps:

1. Read data: we attached an example of raw data in: "data_example.csv".
2. Discard data: we only take into account a window of 4000 frames for the analysis.
3. Normalize data: 
    - Angles: the angles are normalized between [-1, 1]. In order to do this, different trigonometric functions were needed,
as each angle had different relative measures compared with the ground position. Pitch - sin, Yaw - cos, Roll - sin.
    - Position: the values of the X, Y, Z axes are normalized between [0, 1]

4. Compute distance head - controllers. And normalized the distance.
5. Plot the values.
6. Compute angle features (min, max, mean, standard deviation)
7. Compute and plot angle frequencies. (Compute power spectral density)
8. Compute head-controllers features.

Then we put together all the features and after we used these for a classifier.  