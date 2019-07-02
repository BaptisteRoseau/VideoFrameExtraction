LIBS= -lopencv_gapi \
-lopencv_photo \
-lopencv_highgui \
-lopencv_imgcodecs \
-lopencv_stitching \
-lopencv_core \
-lopencv_videoio \
-lopencv_dnn \
-lopencv_video \
-lopencv_imgproc \
-lopencv_ml \
-lopencv_features2d \
-lopencv_objdetect \
-lopencv_flann \
-lstdc++fs \
#-lboost_python \
#-lpython3.6 \
#-L/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu/ \
#-L/usr/lib/python3.6 \
#-lboost_python-py36

INCLUDE= -I /usr/local/include/opencv4/ -I /usr/include/python3.6/

WALL= -std=c++17 -Wall -Wextra -g -Wno-unused-variable -Wno-unused-parameter

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

lib: $(SRC)
	g++ -DNOMAIN -fPIC $(WALL) $(INCLUDE) $(SRC) $(LIBS) -shared -o libextractframe.so

testpy: lib
	python3 test.py

dir: $(OUTPUT)
	./$(OUTPUT) $(DD_DIR_ARGS)

valgrind: $(OUTPUT)
	valgrind --leak-check=full --show-leak-kinds=all ./$(OUTPUT) $(ARGS)

valgrinddir: $(OUTPUT)
	valgrind --leak-check=full --show-leak-kinds=all ./$(OUTPUT) $(DD_DIR_ARGS)

clean:
	rm -rf *.o
	rm $(OUTPUT)