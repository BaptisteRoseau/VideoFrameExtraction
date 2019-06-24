#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio/legacy/constants_c.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/io.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <assert.h>
#include <iostream>
#include <cstring>
#include <random>

using namespace std;
using namespace cv;

#define SKIP_FRAMES 100
#define SKIP_SECONDS 1.2
#define PRINT_INTERVAL 5  // No print: 0
#define MAX_FRAME_IDX 0     // No limit: 0
#define DFLT_MEAN_COUNTED 20
#define THRESHOLD_COEF 0.25
#define SAVE_CHANCE 0.2

//TODO: Un truc d'input propre avec des arguments variables et tout !!
//TODO: Factoriser et mettre dans des fonctions ce qui peut l'être pour simplifier la lecture du main
//TODO: Essayer de boucher les fuites mémoires dûes à OpenCV
//TODO: Equivalent de matplotlib pour C++
//TODO: Ajouter une condition "Si un personnage est sur les 2 frames"
//TODO: BOUCHER LES FUITES MEMOIRES !!!!!!
//TODO: Renommer de façon plus explicite

/*======== SUBFUNCTIONS DECLARATION ==========*/

void skip_frames(VideoCapture &video, const unsigned int nb_frames, unsigned int &frames_index);
void get_next_frame(VideoCapture &video, Mat &prev, Mat &curr, unsigned int &index);
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
	// Macro verification
	//assert(SKIP_FRAMES == 0 || SKIP_SECONDS == 0);
	assert(SKIP_FRAMES >= 0 && SKIP_SECONDS >= 0);
	assert(PRINT_INTERVAL >= 0);
	assert(MAX_FRAME_IDX >= 0);
	assert(DFLT_MEAN_COUNTED > 0);
	assert(THRESHOLD_COEF > 0);

 	// Print usage if not enough parameters
	if (argc != PARAM+1){
		usage(argv[0]);
		return -1;
	}
	
	// Opening Video
    VideoCapture video(argv[1]); // Open the argv[1] video file
    if(!video.isOpened()){
		cerr << "Couldn't open " << argv[1] << endl;
		return -1;
	}

	bool verbose = true;
	// Building directory if doesn't exists
	check_directory(argv[2]);

	// Building prefix for images writing
	char *file      = argv[1];
	string dir      = argv[2];
	char *lastdot   = strrchr(file, '.');
    if (lastdot     != NULL) {*lastdot = '\0';} //Removing extension (ex: .mp4)
	char *lastslash = strrchr(file, '/');
	if (lastslash   != NULL) {file = lastslash;} //Removing previous path (ex: ../)
	string output_prefix = dir+"/"+file+"_frame_";

	// Video settings
	double width     = video.get(CV_CAP_PROP_FRAME_WIDTH);
	double height    = video.get(CV_CAP_PROP_FRAME_HEIGHT);
	double fps       = video.get(CV_CAP_PROP_FPS);
	double nb_frames = video.get(CV_CAP_PROP_FRAME_COUNT);
	if (verbose){
		cout << "Video Properties: " << width << "x" << height
		<< ", " << fps << " fps, " << nb_frames << " frames." << endl;
	}

	// Parameters initialisation
    Mat prev_frame, curr_frame;
	video >> prev_frame; // Getting first frame
	video >> curr_frame; // Getting second frame
	Mat *diff_frame = new Mat(prev_frame.size(), prev_frame.type()); // Buffer for difference picture
	unsigned int curr_frame_idx = 1;
	double diff, diff_coef, mean = 0; // The mean of all computed differences
	unsigned int loop_idx = 0;    // Counter on how many means were calculated
	unsigned int max_frame_idx = MAX_FRAME_IDX == 0 ? nb_frames + 1 : MAX_FRAME_IDX;
	float r;

	// Main Loop
	while(curr_frame_idx < max_frame_idx && !curr_frame.empty()){ // While frames remain
		skip_frames(video, SKIP_FRAMES, curr_frame_idx);
		get_next_frame(video, prev_frame, curr_frame, curr_frame_idx);
		if (curr_frame.empty()){
			break;
		}

		// Displaying script advancement
		if (verbose && PRINT_INTERVAL != 0 && loop_idx % PRINT_INTERVAL == 0){ //FIXME
			cout << "Frame: " << curr_frame_idx
			<< "\tMean: " << mean
			<< "\tMean coef: " << diff_coef << endl;
		}

		// Computing difference between frames
		diff = abs(difference(prev_frame, curr_frame, diff_frame, 10)); // Returns the difference matrix and the value
		//cout << "Difference: " << diff << endl; //Remove me after

		// Ajusting mean
		if (loop_idx == 0){mean = mean + diff;}
		else {mean += (diff - mean)/loop_idx;}

		// If difference is unusually high, write the picture into the given directory
		diff_coef = abs((diff - mean))/mean;
		//cout << mean << "\t" << diff << ":\t" << diff_coef << endl;

		r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		if (r <= SAVE_CHANCE
		 && loop_idx > DFLT_MEAN_COUNTED
		 && (diff_coef < THRESHOLD_COEF)){
			if (verbose){
				cout << "Difference Ratio: " << diff_coef << " at frames " << curr_frame_idx-SKIP_FRAMES << " " << curr_frame_idx << endl;
			}
			write_frames(prev_frame, curr_frame, output_prefix, curr_frame_idx);
			get_next_frame(video, prev_frame, curr_frame, curr_frame_idx);
		}

		loop_idx++;
	}

	// Cleaning memory
	prev_frame.release();
	curr_frame.release();
  	video.release();
	delete diff_frame;

	if (verbose){
		cout << "Exited successfully." << endl;
	}
    return 0;
}


/*======== SUBFUNCTIONS IMPLEMENTATION ==========*/

void get_next_frame(VideoCapture &video, Mat &prev, Mat &curr, unsigned int &index){
	prev.release();
	curr.copyTo(prev);
	curr.release();
	video >> curr;
	index++;
}

void write_frames(const Mat &prev, const Mat &next, string prefix, unsigned int index){
	bool a = imwrite(prefix+to_string(index-SKIP_FRAMES)+"_IN.jpg", prev);
	bool b = imwrite(prefix+to_string(index)+"_OUT.jpg", next);
	if (a && b){
		cout << "Wrote frames " << index-SKIP_FRAMES << " " << index
		<< " as "+prefix+to_string(index-SKIP_FRAMES)+".jpg" << endl;
	} else {
		cerr << "Failed to write frames " << index-SKIP_FRAMES << " "
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
}

void skip_frames(VideoCapture &video, const unsigned int nb_frames, unsigned int &frames_index){
	Mat *tmp = new Mat();
	for (unsigned int i = 0; i < nb_frames; i++){
		video >> *tmp;
		frames_index++;
		tmp->release();
	}
}


/*======== DIFFERENCE FUNCTION IMPLEMENTATION ==========*/


typedef cv::Point3_<uchar> Pixel;
class Operator{ // An operator used by Mat::forEach()
	private:
		Operator();
		const Mat *prev;
		const Mat *next;
		int halfside;
		unsigned int diff_len;
		double *diff; // Buffer for difference calculus

	public:
		Operator(const Mat *prev, const Mat *next, const unsigned int side){
			assert(prev->dims == next->dims && prev->size() == next->size() && prev->channels() == next->channels());
			this->prev     = prev;
			this->next     = next;
			this->halfside = (int) side/2;
			this->diff_len = next->rows*next->cols;
			this->diff     = new double[this->diff_len]; // One case for each computed pixel to prevent parallel issue
		}

		~Operator(){
			//delete[] this->diff; //Segfault here
		}

		// Local difference operation (called for each pixel)
		void operator()(Pixel &pixel, const int * pos) const{
			// Submatrix extraction around current pixel (O(1))
			Range r_x = Range(max(pos[0]-halfside, 0), min(pos[0]+halfside+1, this->prev->rows));
			Range r_y = Range(max(pos[1]-halfside, 0), min(pos[1]+halfside+1, this->prev->cols));
			const Mat sub_prev = this->prev->operator()(r_x, r_y);
			const Mat sub_next = this->next->operator()(r_x, r_y);

			// Submatrix difference calculation
			Scalar sum_diff = sum(sub_next) - sum(sub_prev); // Difference on each channel
			this->diff[pos[0]*prev->cols + pos[1]] = sum(sum_diff)[0] / (sub_prev.rows * sub_prev.cols); // Total difference ratio

			//TODO: Set the pixel value according to the diff (care about double -> uchar)
			//TODO: Apply a log function to reduce the probability of having a 'nan'
			//TODO: Find a "fast" log approximation in c++
		}

		// Retrieve difference value
		// /!\ Call this after the forEach function, not before !!
		double getDiff(){
			double ret = 0;
			for (unsigned int i = 0; i < this->diff_len; i++){
				ret += abs(this->diff[i]);
			}
			return ret;
		}
};

//TODO: Une autre fonction difference pour laisser le choix à l'utilisateur de stocker ou non la matrice "différence";
double difference(const Mat &prev, const Mat &next, Mat *buffer, const unsigned int area_side = 5){
	//Note: Buffer are used because of the "const" qualifier of Operator::operator() requiered by Mat::forEach
	Operator op = Operator(&prev, &next, area_side);
	buffer->forEach<Pixel>(op);
	return op.getDiff();
}

/*
USER-FRIENDLY:
	- Adust macros (give them as argument)
	- Add the possibility to add a constraint function
	- Python Interface
	- 
 */
