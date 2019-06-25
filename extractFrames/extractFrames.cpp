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
#include <stack>
#include <fstream>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = filesystem;

#define DISPLAY(stream) if (verbose){(stream);}

//TODO: Essayer de boucher les fuites mémoires dûes à OpenCV
//TODO: BOUCHER LES FUITES MEMOIRES !!!!!!
//TODO: Renommer de façon plus explicite

/*
USER-FRIENDLY:
	- Python Interface
 */

/*======== SUBFUNCTIONS IMPLEMENTATION ==========*/

double difference(const Mat &prev, const Mat &next, const unsigned int area_side);

void get_next_frame(VideoCapture &video, Mat &prev, Mat &curr){
	prev.release();
	curr.copyTo(prev);
	curr.release();
	video >> curr;
}

void write_frame(const Mat &m, string dir, string name, bool verbose){
	string tmp = dir+'/'+name;
	bool a = imwrite(tmp, m);
	if (a){
		DISPLAY(cout << "Wrote frame " << tmp << endl);
	}
	else{
		DISPLAY(cout << "Failed to write frame " << tmp << endl);
	}
}

void build_output_dir(const char *path){
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

void video_skip_frames(VideoCapture &video, const unsigned int nb_frames, unsigned int &frames_index){
	bool success = true;
	for (unsigned int i = 0; success && i < nb_frames; i++){
		success = video.grab();
		frames_index++;
	}
}

/* void get_filename(const char *file, char *buffer){ //FIXME: modifer l'adresse du buffer qui est alloué avec new c'est pas une bonne idée
	strcpy(buffer, file);
	char *lastdot   = strrchr(buffer, '.');
    if (lastdot    != NULL) {*lastdot = '\0';} //Removing extension (ex: .mp4)
	char *lastslash = strrchr(buffer, '/');
	if (lastslash  != NULL) {buffer = lastslash+1;} //Removing previous path (ex: ../)
} */

string get_filename(const string &filepath){
	return fs::path(filepath).filename().replace_extension("");
}

void str_normalize(string &s){
	for (size_t i = 0; i < s.length(); i++){
		switch (s[i]){
			case ' ': s[i] = '_'; break;
			case '(': s[i] = '_'; break;
			case ')': s[i] = '_'; break;
			case '[': s[i] = '_'; break;
			case ']': s[i] = '_'; break;
		}
	}
}

bool is_supported_videofile(const fs::path path){
	if (!path.has_extension()){
		return false;
	}
	string ext = path.extension();
	 //TODO: Trouver les extensions supportées et les ajouter ici
	return true;
}

stack<string> *get_video_files(const char *in_path, double file_proportion, bool verbose){
	stack<string> *vid_files = new stack<string>();
	fs::directory_entry f = fs::directory_entry(in_path);

	// File
	if (f.is_regular_file() && is_supported_videofile(f.path())){
		vid_files->push((string) f.path());
		DISPLAY(cout << "Added: " << f.path() << '\n');
		return vid_files;
	}

	// Directory
	if (f.is_directory()){
		double r;
		for(auto& p: fs::recursive_directory_iterator(in_path)){
			if (p.is_regular_file() && is_supported_videofile(p.path())){
				r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
				if (r < file_proportion){
					vid_files->push((string) p.path());
					DISPLAY(cout << "Added: " << p.path() << '\n');
				}
			}
		}
		return vid_files;
	}

	// Error case
	cerr << "Fatal Error: " << in_path << " is nor a file nor a directory." << endl;
	exit(EXIT_FAILURE);
}



/**
 * @brief 
 * 
 * @param in_path 
 * @param out_dir 
 * @param skip_frames 
 * @param skip_seconds 
 * @param stop_at_frame 
 * @param display_interval 
 * @param save_in_between_frames 
 * @param compute_difference 
 * @param diff_threshhold 
 * @param pic_save_proba 
 * @param verbose 
 * @param min_mean_couted 
 * @param first_frame_func 
 * @param second_frame_func 
 * @return int 
 */
int process(const char *in_path, const char *out_dir,
			unsigned int skip_frames            = 3,
			double       skip_seconds           = 0,
			unsigned int stop_at_frame          = 1000,
			unsigned int display_interval       = 10,
			bool         save_in_between_frames = false,
			bool         compute_difference     = false,
			double       diff_threshhold        = 0.2,
			double       pic_save_proba         = 0.1,
			bool         verbose                = true,
			double       min_mean_couted        = 20,
			double       file_proportion        = 0.01,
			bool (*first_frame_func)(const Mat &)  = NULL,
			bool (*second_frame_func)(const Mat &) = NULL){ //TODO: Ajouter la possibilité de ne pas créer de sous-repertoires et juste stocker les images
	
	// Parameters verification
	assert(skip_frames == 0 || skip_seconds == 0);
	assert(skip_frames != 0 || skip_seconds != 0);
	assert(skip_seconds >= 0);
	assert(diff_threshhold >= 0);
	assert(0 <= pic_save_proba && pic_save_proba <= 1);
	assert(0 <= file_proportion && file_proportion <= 1);

	// Retrieving video file paths into vid_path_stack
	stack<string> *vid_path_stack = get_video_files(in_path, file_proportion, verbose);

	// Building output directory if doesn't exists
	build_output_dir(out_dir);
	string str_out_dir = out_dir;

	while (!vid_path_stack->empty()){
		// Retrieving video path
		string filepath = vid_path_stack->top();
		vid_path_stack->pop();
		string filename = get_filename(filepath);
		str_normalize(filename);

		// Opening Video
		VideoCapture video(filepath); // Open the argv[1] video file
		if(!video.isOpened()){
			cerr << "Couldn't open " << filepath << endl;
			continue;
		}

		// Building prefix for images writing
		string curr_file_out_dir = str_out_dir+"/"+filename;
		fs::directory_entry dir_entry = fs::directory_entry(curr_file_out_dir);
		
		// Video settings
		double width     = video.get(CV_CAP_PROP_FRAME_WIDTH);
		double height    = video.get(CV_CAP_PROP_FRAME_HEIGHT);
		double fps       = video.get(CV_CAP_PROP_FPS);
		double nb_frames = video.get(CV_CAP_PROP_FRAME_COUNT);
		DISPLAY(cout << "Loaded " << filename << "\nVideo Properties: " << width << "x" << height
			<< ", " << fps << " fps, " << nb_frames << " frames." << endl);

		// Arguments interpretation
		unsigned int _skip_frames   = skip_frames == 0 ? skip_seconds*fps : skip_frames;
		unsigned int _stop_at_frame = stop_at_frame == 0 ? nb_frames + 1 : stop_at_frame;

		// Parameters initialisation
		Mat prev_frame, curr_frame;
		video >> prev_frame; // Getting first frame
		video >> curr_frame; // Getting second frame
		unsigned int prev_frame_idx = 0;
		unsigned int curr_frame_idx = 1;
		unsigned int loop_idx = 0;    // Counter on how many means were calculated
		double diff, diff_coef, mean = 0; // The mean of all computed differences
		float r;
		bool fff_verified, sff_verified;
		queue<unsigned int> in_btw_frm_indexes;

		// Main Loop
		while(curr_frame_idx < _stop_at_frame && !curr_frame.empty()){ // While frames remain
			// Getting next frame
			prev_frame_idx = curr_frame_idx;
			video_skip_frames(video, _skip_frames, curr_frame_idx);
			get_next_frame(video, prev_frame, curr_frame);
			curr_frame_idx++;

			if (curr_frame.empty()){
				break;
			}

			// Displaying script advancement
			if (display_interval != 0 && loop_idx % display_interval == 0){
				if (!compute_difference){
					DISPLAY(cout << "Frame: " << curr_frame_idx << endl);
				} else {
					DISPLAY(cout << "\tMean: " << mean << "\tMean coef: " << diff_coef);
				}
			}

			// Computing difference between frames
			if (compute_difference){
				diff = abs(difference(prev_frame, curr_frame, 10)); // Returns the difference matrix and the value
			}

			// Ajusting mean
			if (compute_difference){
				if (loop_idx == 0){mean = mean + diff;}
				else {mean += (diff - mean)/loop_idx;}

				// If difference is unusually high, write the picture into the given directory
				diff_coef = abs((diff - mean))/mean;
			}

			// Apply user custom restrictions
			fff_verified = first_frame_func == NULL  ? true : first_frame_func(prev_frame);
			sff_verified = second_frame_func == NULL ? true : second_frame_func(curr_frame);

			// If everything is ok, save the picture
			r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
			if (r <= pic_save_proba
			&& fff_verified
			&& sff_verified
			&& (!compute_difference
			|| (loop_idx > min_mean_couted
				&& diff_coef < diff_threshhold))){
				// Displaying information about difference if needed
				if (compute_difference ){
					DISPLAY(cout << "Difference Ratio: " << diff_coef << " at frames " << curr_frame_idx-_skip_frames << " " << curr_frame_idx << endl);
				}
				// Creating directory if doesn't exist
				if (!dir_entry.exists()){
					if (!fs::create_directory(dir_entry.path())){
						DISPLAY(cout << "Failed to create directory " << dir_entry.path()
						<< "\nSkipping to next video source." << endl);
						break;
					}
				}
				// Writing frames into the directory
				write_frame(prev_frame, curr_file_out_dir, filename+"_frame_"+to_string(prev_frame_idx)+"_IN.jpg", verbose);
				write_frame(curr_frame, curr_file_out_dir, filename+"_frame_"+to_string(curr_frame_idx)+"_OUT.jpg", verbose);
				// Saving indexes to save the in-between frames if needed
				if (save_in_between_frames){
					in_btw_frm_indexes.push(prev_frame_idx);
					in_btw_frm_indexes.push(curr_frame_idx);
				}
				get_next_frame(video, prev_frame, curr_frame);
			}

			loop_idx++;
		}

		if (save_in_between_frames){
			//TODO: Reparcourir la video
		}

		// Cleaning memory
		prev_frame.release();
		curr_frame.release();
		video.release();
	}

	delete vid_path_stack;

	DISPLAY(cout << "Exited successfully." << endl);

	return EXIT_SUCCESS;
}


/*======== DIFFERENCE FUNCTION IMPLEMENTATION ==========*/

//FIXME

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
			/* this->diff[pos[0]*prev->cols + pos[1]] = 
				abs(sum(sum(sub_next))[0] - sum(sum(sub_prev))[0])
				/ (sub_prev.rows * sub_prev.cols); // Total difference ratio */

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

double difference(const Mat &prev, const Mat &next, const unsigned int area_side = 5){
	//Note: Buffer are used because of the "const" qualifier of Operator::operator() requiered by Mat::forEach
	Operator op = Operator(&prev, &next, area_side);
	prev.forEach<Pixel>(op);
	return op.getDiff();
}


/* PYTHON */

//TODO: Macros "boost" pour python

/*======== MAIN ==========*/

#ifndef NOMAIN
void usage(char* name){
	cout << "Usage: " << name << " <path to video> <path write directory>" << endl;
}

#define PARAM 2
int main(int argc, char** argv)
{
	if (argc != PARAM+1){
		usage(argv[0]);
		return -1;
	}
	process(argv[1], argv[2]);
    return 0;
}
#endif