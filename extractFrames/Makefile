LIBS= -lopencv_gapi\
-lopencv_photo\
-lopencv_highgui\
-lopencv_imgcodecs\
-lopencv_stitching\
-lopencv_core\
-lopencv_videoio\
-lopencv_imgproc\
-lopencv_dnn\
-lopencv_video\
-lopencv_imgproc\
-lopencv_ml\
-lopencv_features2d\
-lopencv_objdetect\
-lopencv_flann

INCLUDE= -I /usr/local/include/opencv4/ extractFrames.cpp

WALL= -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter

all:
	g++ $(WALL) $(INCLUDE) $(LIBS)

nv:
	nvcc $(WALL) $(INCLUDE) $(LIBS)
