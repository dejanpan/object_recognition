Object Recognition
==================

Framework to recognize objects on Kinect depth images. This project is in early development stage.

You can install the dependencies on ubuntu by executing: `sudo apt-get install mayavi2 python-numpy python-scipy python-matplotlib`
It also requires python opencv module, so if it is not installed (for example from ROS) install it using `sudo apt-get install python-opencv`

Right now only simple syntetic test is availible:

1. Go to the object_recognition directory.

2. Create directories for the generated data `mkdir training_data test_data`

3. Generate random training and test set: `./generate_random_scenes.py models/ training_data/ 1000` `./generate_random_scenes.py models/ test_data/ 100`

4. Train a classifier: `./train.py`

5. Test classifier: `./classify.py`

6. It will test computed classifier and show misclassified scenes.
