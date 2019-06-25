LIBS= -lopencv_gapi\
-lopencv_photo\
-lopencv_highgui\
-lopencv_imgcodecs\
-lopencv_stitching\
-lopencv_core\
-lopencv_videoio\
-lopencv_dnn\
-lopencv_video\
-lopencv_imgproc\
-lopencv_ml\
-lopencv_features2d\
-lopencv_objdetect\
-lopencv_flann\
-lstdc++fs

INCLUDE= -I /usr/local/include/opencv4/

WALL= -std=c++17 -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -g

SRC=extractFrames.cpp
OUTPUT=extractFrames
ARGS= ./AccelWorld1.mp4 testDir

DD_DIR_ARGS=/media/broseau/BR_MULTIMEDIA/Disque\ Dur/Vidéos/Animes testDir

$(OUTPUT): $(SRC)
	g++ $(WALL) $(INCLUDE) $(SRC) $(LIBS) -o $(OUTPUT)

nv: $(SRC)
	nvcc $(WALL) $(INCLUDE) $(SRC) $(LIBS)

run: $(OUTPUT)
	./$(OUTPUT) $(ARGS)

gdb: $(OUTPUT)
	gdb -ex run --args $(OUTPUT) $(ARGS)

lib:
	echo TODO avec -NOMAIN

dir: $(OUTPUT)
	./$(OUTPUT) $(DD_DIR_ARGS)

valgrind: $(OUTPUT)
	valgrind --leak-check=full ./$(OUTPUT) $(ARGS)

valgrinddir: $(OUTPUT)
	valgrind --leak-check=full ./$(OUTPUT) $(DD_DIR_ARGS)

clean:
	rm -rf *.o
	rm $(OUTPUT)