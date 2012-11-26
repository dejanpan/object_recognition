Object Recognition
==================

Framework to recognize objects on Kinect depth images. This project is in early development stage.

You can install the dependencies on ubuntu by executing:
sudo apt-get install libopencv-dev mayavi2 python-numpy python-scipy python-matplotlib

Right now only simple syntetic test is availible:

Go to the object_recognition directory.

Create directories for the generated data:
mkdir training_data test_data

Generate random training and test set:
./generate_random_scenes.py models/ training_data/ 1000
./generate_random_scenes.py models/ test_data/ 100

Train a classifier:
./train.py

Test classifier:
./classify.py

It will test computed classifier and show misclassified scenes.
