#include <opencv2/opencv.hpp>
#include <opencv2/videoio/legacy/constants_c.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/io.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <iostream>
#include <cstring>

using namespace std;
using namespace cv;


#define SKIP_FRAMES 1
#define SKIP_SECONDS 1.2
#define PRINT_INTERVAL 5  // No print: 0
#define MAX_FRAME_IDX 15
#define DFLT_MEAN_COUNTED 10
#define THRESHOLD_COEF 2
#define WRITE true

//TODO: Un truc d'input propre avec des arguments variables et tout !!
//TODO: Factoriser et mettre dans des fonctions ce qui peut l'être pour simplifier la lecture du main
//TODO: Essayer de boucher les fuites mémoires dûes à OpenCV
//TODO: Equivalent de matplotlib pour C++
//TODO: Ajouter une condition "Si un personnage est sur les 2 frames"

/*======== SUBFUNCTIONS DECLARATION ==========*/

void skip_frames(VideoCapture &video, const unsigned int nb_frames, unsigned int &frames_index);
double difference(const Mat &prev, const Mat &next, Mat *buffer, const unsigned int area_side);
void write_frames(const Mat &prev, const Mat &next, string prefix, unsigned int index);
void check_directory(const char *path);

/*======== MAIN ==========*/

void usage(char* name){
	cout << "Usage: " << name << " <path to video> <path write directory>" << endl;
}

#define PARAM 2
int main(int argc, char** argv)
{
	// Marco verification
	//assert(SKIP_FRAMES == 0 || SKIP_SECONDS == 0);
	assert(SKIP_FRAMES >= 0 && SKIP_SECONDS >= 0);
	assert(PRINT_INTERVAL >= 0);
	assert(MAX_FRAME_IDX > 0);
	assert(DFLT_MEAN_COUNTED > 0);
	assert(THRESHOLD_COEF > 0);
	//assert(typeof(WRITE) == bool);

	if (argc != PARAM+1){ // Print usage if not enough parameters
		usage(argv[0]);
		return -1;
	}
	
	// Opening Video
    VideoCapture video(argv[1]); // Open the argv[1] video file
    if(!video.isOpened()){
		cerr << "Couldn't open " << argv[1] << endl;
		return -1;
	}

	// Building directory if doesn't exists
	check_directory(argv[2]);

	// Building prefix for images writing
	char *file = argv[1];
	string dir  = argv[2];
	char *lastdot = strrchr(file, '.');
    if (lastdot != NULL) {*lastdot = '\0';} //Removing extension (ex: .mp4)
	char *lastslash = strrchr(file, '/');
	if (lastslash != NULL) {file = lastslash;} //Removing previous path (ex: ../)
	string output_prefix = dir+"/"+file+"_frame_";

	// Video settings
	double width     = video.get(CV_CAP_PROP_FRAME_WIDTH);
	double height    = video.get(CV_CAP_PROP_FRAME_HEIGHT);
	double fps       = video.get(CV_CAP_PROP_FPS);
	double nb_frames = video.get(CV_CAP_PROP_FRAME_COUNT);
	cout << "Video Properties: " << width << "x" << height
	 << ", " << fps << " fps, " << nb_frames << " frames." << endl;

	// Parameters initialisation
    Mat prev_frame, curr_frame;
	video >> prev_frame; // Getting first frame
	video >> curr_frame; // Getting second frame
	Mat *diff_frame = new Mat(prev_frame.size(), prev_frame.type()); // Buffer for 
	unsigned int curr_frame_idx = 1;
	double diff, diff_coef, mean = 0; // The mean of all computed differences
	unsigned int mean_counter = 0; // Counter on how many means were calculated

	// Main Loop
	while(curr_frame_idx < MAX_FRAME_IDX && !curr_frame.empty()){ // While frames remain
		skip_frames(video, SKIP_FRAMES, curr_frame_idx);

		// Displaying script advancement
		if (PRINT_INTERVAL != 0 && curr_frame_idx % PRINT_INTERVAL == 0){
			cout << "Frame: " << curr_frame_idx
			<< "\tMean: " << mean
			<< "\tMean coef: " << diff_coef << endl;
		}

		// Computing difference between frames
		diff = difference(prev_frame, curr_frame, diff_frame, 5); // Returns the difference matrix and the value

		// Ajusting mean and writing pictures into the directory
		if (mean_counter == 0){ mean = mean + diff; mean_counter++;}
		else {mean += (diff - mean)/mean_counter; mean_counter++;}

		// If difference is unusually high, write the picture into the given directory
		diff_coef = (diff - mean)/mean;
		if (WRITE
		 && curr_frame_idx > DFLT_MEAN_COUNTED
		 && mean != 0 
		 && (diff_coef < -THRESHOLD_COEF
		  || THRESHOLD_COEF < diff_coef)){
			cout << "\tDifference: " << diff_coef << " at frames " << curr_frame_idx-1 << " " << curr_frame_idx << endl;
			write_frames(prev_frame, curr_frame, output_prefix, curr_frame_idx);
		}

		// Saving previous frame
		prev_frame.release();
		prev_frame = curr_frame;
	}

	// Cleaning memory
  	video.release();
	delete diff_frame;
	
	
  	// Closes all the frames
	cout << "Exited successfully." << endl;
    return 0;
}

/*======== SUBFUNCTIONS IMPLEMENTATION ==========*/

void write_frames(const Mat &prev, const Mat &next, string prefix, unsigned int index){
	if (imwrite(prefix+to_string(index-1)+".jpg", prev)
	&& imwrite(prefix+to_string(index)+".jpg", next)){
		cout << "Wrote frames " << index-1 << " " << index
		<< " as "+prefix+to_string(index-1)+".jpg" << endl;
	} else {
		cerr << "Failed to write frames " << index-1 << " "
		<< index << "as" << prefix << endl;
	}
}

void check_directory(const char *path){
	struct stat info;
	if(stat(path, &info) != 0){
		if (mkdir(path, S_IRWXU) != 0){
			cerr << "Fatal Error: Couldn't create " << path << endl;
			exit(EXIT_FAILURE);
		}
	}
	else if (!S_ISDIR(info.st_mode)){ 
		cerr << "Fatal Error: " << path << " is not a directory" << endl;
		exit(EXIT_FAILURE);
	}
	// else, directory already exists
}

void skip_frames(VideoCapture &video, const unsigned int nb_frames, unsigned int &frames_index){
	Mat tmp;
	for (unsigned int i = 0; i < nb_frames; i++){
		video >> tmp;
		frames_index++;
		tmp.release();
	}
}

double difference(const Mat &prev, const Mat &next, Mat *buffer, const unsigned int area_side = 5){
	

	return 10;
}
