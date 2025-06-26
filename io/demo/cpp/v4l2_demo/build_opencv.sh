#!/bin/bash

g++ capture_opencv.cpp -g -o capture_app \
-I/usr/local/include/opencv4 \
-L/usr/local/lib \
-lopencv_core -lopencv_imgproc -lopencv_highgui \
-I/usr/include/libcamera -lcamera -lcamera-base