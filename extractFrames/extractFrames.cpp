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
#include <limits.h>

using namespace std;
using namespace cv;
namespace fs = filesystem;

#define DISPLAY(stream) if (verbose){stream;}
#define BET_FRAMES_DIRNAME "in_between"
#define SAME_FRAME_THRESHOLD 5

//TODO: BOUCHER LES FUITES MEMOIRES !!!!!! (Il en reste un peu mais sans plus)
//TODO: Un autre script avec une condition que sur 1 image ?
//TODO: Renommer de façon plus explicite
//TODO: Factoriser !!

/*======== SUBFUNCTIONS IMPLEMENTATION ==========*/

/* String and directory tools */
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
			if (p.is_regular_file()){// && is_supported_videofile(p.path())){
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
	cerr << "Fatal Error: " << in_path << " is not a file nor a directory." << endl;
	exit(EXIT_FAILURE);
}

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

/* Frame Management tools */
double difference(const Mat &prev, const Mat &next, const unsigned int area_side);

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

bool are_identic_frames(const Mat &m1, const Mat &m2){
	assert(m1.dims       == m2.dims 
		&& m1.size()     == m2.size() 
		&& m1.channels() == m2.channels());
	Mat diff = m1 - m2;
	double minVal, maxVal;
	Point minLoc, maxLoc;
	cout << "Empty: " << diff.empty() << endl;
	minMaxLoc(diff, &minVal, &maxVal, &minLoc, &maxLoc);
	return maxVal < SAME_FRAME_THRESHOLD;
}

void get_next_frame(VideoCapture &video, Mat &prev, Mat &curr){
	prev.release();
	curr.copyTo(prev);
	curr.release();
	video >> curr;
}

void video_skip_frames(VideoCapture &video, const unsigned int nb_frames){
	bool success = true;
	for (unsigned int i = 1; success && i < nb_frames; i++){
		success = video.grab();
	}
}

void video_skip_frames_stock(VideoCapture &video, const unsigned int nb_frames, queue<Mat*> &in_btw_frm_stocked, bool remove_identic_frames){
	if (!remove_identic_frames){
		Mat *tmp_mat;
		for (unsigned int i = 1; i < nb_frames; i++){
			tmp_mat = new Mat();
			video >> (*tmp_mat);
			if (tmp_mat->empty()){ break;}
			in_btw_frm_stocked.push(tmp_mat);
		}
	} else {
		Mat *tmp_mat;
		Mat *tmp_mat_prev = new Mat(); //TODO: Initialize tmp_mat_prev
		for (unsigned int i = 1; i < nb_frames; i++){
			tmp_mat = new Mat();
			video >> (*tmp_mat);
			if (tmp_mat->empty()){ break;}
			cout << are_identic_frames(*tmp_mat_prev, *tmp_mat);
			if (!tmp_mat_prev->empty()
				&& !are_identic_frames(*tmp_mat_prev, *tmp_mat)){
				in_btw_frm_stocked.push(tmp_mat);
			}
			tmp_mat_prev->release();
			*tmp_mat_prev = tmp_mat->clone();
		}
		delete tmp_mat_prev;
	}
}

void empty_frame_stock(queue<Mat*> &in_btw_frm_stocked){
	while(!in_btw_frm_stocked.empty()){
		delete in_btw_frm_stocked.front();
		in_btw_frm_stocked.pop();
	}
}



//TODO: doc
/**
 * @brief 
 * 
 * @param in_path 
 * @param out_dir 
 * @param skip_frames 
 * @param skip_seconds 
 * @param stop_at_frame 
 * @param display_interval 
 * @param min_mean_counted 
 * @param save_in_between_frames 
 * @param stock_in_between_frames 
 * @param remove_identic_frames 
 * @param compute_difference 
 * @param verbose 
 * @param diff_threshhold 
 * @param pic_save_proba 
 * @param file_proportion 
 * @param timeout 
 * @param first_frame_func 
 * @param second_frame_func 
 * @return int 
 */
int exctractFrames(const char *in_path, const char *out_dir,
			unsigned int skip_frames             = 0,
			double       skip_seconds            = 0,
			unsigned int stop_at_frame           = 0,
			unsigned int display_interval        = 1,
			unsigned int min_mean_counted        = 20,
			bool         save_in_between_frames  = true,
			bool         stock_in_between_frames = true,
			bool         remove_identic_frames   = false,  //TODO: Tester
			bool         compute_difference      = false, //TODO: Boucher la fuite memoire
			bool         verbose                 = true,
			double       diff_threshhold         = 1,
			double       pic_save_proba          = 1,
			double       file_proportion         = 1,
			double       timeout                 = 0,     //In seconds
			bool (*first_frame_func)(const Mat &)  = NULL,
			bool (*second_frame_func)(const Mat &) = NULL){

	// Setting default skip frames to 1
	if (skip_frames == 0 && skip_seconds == 0){
		skip_frames = 1;
	}

	// Parameters verification
	assert(skip_frames == 0 || skip_seconds == 0);
	assert(skip_frames != 0 || skip_seconds != 0);
	assert(skip_seconds >= 0);
	assert(timeout >= 0);
	assert(diff_threshhold >= 0);
	assert(0 <= pic_save_proba && pic_save_proba <= 1);
	assert(0 <= file_proportion && file_proportion <= 1);
	//TODO: Ajouter des verfication

	// Retrieving video file paths into vid_path_stack
	stack<string> *vid_path_stack = get_video_files(in_path, file_proportion, verbose);

	// Building output directory if doesn't exists
	build_output_dir(out_dir);
	string str_out_dir = out_dir;

	// Global parameter initialisation
	unsigned int global_counter = 0;
	srand(time(NULL));
	time_t t0 = time(NULL);

	while (!vid_path_stack->empty()){
		// Retrieving video path
		string filepath = vid_path_stack->top();
		vid_path_stack->pop();
		string filename = get_filename(filepath);
		//str_normalize(filename);

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
		DISPLAY(cout << "\nLoaded " << filename << "\nVideo Properties: " << width << "x" << height
			<< ", " << fps << " fps, " << nb_frames << " frames." << endl);

		// Arguments interpretation
		unsigned int _skip_frames   = skip_frames == 0 ? max(1., skip_seconds*fps) : skip_frames;
		unsigned int _stop_at_frame = stop_at_frame == 0 ? nb_frames + 1 : stop_at_frame;
		double _timeout = timeout == 0 ? DBL_MAX : timeout; 
		if (_skip_frames == 1 && save_in_between_frames){
			DISPLAY(cout << "Error: Cannot save in-between frame wile skiping only 1 frame.\n\
			No in-between frame will be saved.")
			sleep(5);
		}

		// Parameters initialisation
		Mat prev_frame, curr_frame;
		video >> prev_frame; // Getting first frame
		video >> curr_frame; // Getting second frame
		unsigned int curr_frame_idx = video.get(CV_CAP_PROP_POS_FRAMES);;
		unsigned int prev_frame_idx = curr_frame_idx - 1;
		unsigned int loop_idx = 0;        // Also used for as mean counter
		double diff, diff_coef, mean = 0; // The mean of all computed differences
		float r;
		bool fff_verified, sff_verified;
		queue<unsigned int> in_btw_frm_indexes;
		queue<Mat*> in_btw_frm_stocked;
		string frame_path;

		// Main Loop
		while(curr_frame_idx < _stop_at_frame && !curr_frame.empty()){ // While frames remain
			// Getting next frame
			prev_frame_idx = curr_frame_idx;
			if (save_in_between_frames && stock_in_between_frames){
				video_skip_frames_stock(video, _skip_frames, in_btw_frm_stocked, remove_identic_frames);
			} else {
				video_skip_frames(video, _skip_frames);
			}
			curr_frame_idx = video.get(CV_CAP_PROP_POS_FRAMES);
			if (curr_frame.empty()){
				break;
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

			// Displaying script advancement
			if (display_interval != 0 && loop_idx % display_interval == 0){
				if (!compute_difference){
					DISPLAY(cout << "Frame: " << curr_frame_idx
					<< format(" (%.2f %%)", min(100., ((double) 100*curr_frame_idx)/_stop_at_frame)) << endl);
				} else {
					DISPLAY(cout << "Frame: " << curr_frame_idx
					<< format(" (%.2f %%)", min(100., ((double) 100*curr_frame_idx)/_stop_at_frame)) <<
					"\tMean: " << mean << "\tMean coef: " << diff_coef << endl);
				}
			}

			// Chance to save this frame
			r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);

			// Apply user custom restrictions only if necessary
			if (r <= pic_save_proba){
				// If identic frames need to be removed
				if (remove_identic_frames){
					if (are_identic_frames(prev_frame, curr_frame)){
						fff_verified = false;
					} else {
						fff_verified = first_frame_func == NULL  ? true : first_frame_func(prev_frame);
						sff_verified = second_frame_func == NULL ? true : second_frame_func(curr_frame);
					}
				}
				// If identic frames don't need to be removed
				else {
					fff_verified = first_frame_func == NULL  ? true : first_frame_func(prev_frame);
					sff_verified = second_frame_func == NULL ? true : second_frame_func(curr_frame);
				}
			// If the probability to save this picture is too low, don't save it
			} else {
				fff_verified = false;
			}
			
			if (fff_verified
			&& sff_verified
			&& (!compute_difference
			|| (loop_idx > min_mean_counted
				&& diff_coef < diff_threshhold))){
				// Displaying information about difference if needed
				if (compute_difference){
					DISPLAY(cout << "Difference Ratio: " << diff_coef 
					<< " at frames " << curr_frame_idx-_skip_frames 
					<< " " << curr_frame_idx << endl);
				}

				// Creating directory if doesn't exist
				frame_path = out_dir+(string) "/"+to_string(global_counter);
				dir_entry = fs::directory_entry(frame_path);
				if (dir_entry.exists()){
					DISPLAY(cout << "Removing " << dir_entry.path() << endl);
					fs::remove_all(dir_entry.path());
				}
				if (!fs::create_directory(dir_entry.path())){
					DISPLAY(cerr << "Failed to create directory " << dir_entry.path() << endl);
				}

				// Writing frames into the directory
				write_frame(prev_frame, frame_path, "frame_"+to_string(prev_frame_idx)+"_IN.jpg", verbose);
				write_frame(curr_frame, frame_path, "frame_"+to_string(curr_frame_idx)+"_OUT.jpg", verbose);

				// Writing stocked in-between frames
				if (save_in_between_frames && stock_in_between_frames){
					// Creating in-between frames directory
					frame_path = out_dir+(string) "/"
								+to_string(global_counter)
								+(string) "/"
								+(string) BET_FRAMES_DIRNAME;
					dir_entry = fs::directory_entry(frame_path);
					if (dir_entry.exists()){
						DISPLAY(cout << "Removing " << dir_entry.path() << endl);
						fs::remove_all(dir_entry.path());
					}
					if (!fs::create_directory(dir_entry.path())){
						DISPLAY(cerr << "Failed to create directory " << dir_entry.path() << endl);
					}
					// Writing frames into the directory
					unsigned int tmp_idx = prev_frame_idx+1;
					Mat *tmp_mat;
					Mat *tmp_mat_prev = new Mat();
					while(!in_btw_frm_stocked.empty()){
						tmp_mat = in_btw_frm_stocked.front();
						in_btw_frm_stocked.pop();
						if (!remove_identic_frames
						|| (remove_identic_frames
							&& !tmp_mat_prev->empty()
							&& !are_identic_frames(*tmp_mat_prev, *tmp_mat))){
							write_frame(*tmp_mat, frame_path, "frame_"+to_string(tmp_idx)+".jpg", false);
						}
						tmp_idx++;
						tmp_mat_prev->release();
						*tmp_mat_prev = tmp_mat->clone();
						delete tmp_mat;
					}
					delete tmp_mat_prev;
					DISPLAY(cout << format("Wrote frame %u to %u into ", prev_frame_idx+1, tmp_idx)
					 << frame_path << endl;)
				}

				// Saving indexes to save the in-between frames if needed
				if (save_in_between_frames && !stock_in_between_frames){
					in_btw_frm_indexes.push(global_counter);
					in_btw_frm_indexes.push(prev_frame_idx);
					in_btw_frm_indexes.push(curr_frame_idx);
				}

				global_counter++;
				get_next_frame(video, prev_frame, curr_frame);
			}

			empty_frame_stock(in_btw_frm_stocked);
			loop_idx++;

			// Cleaning memory and exiting program if timeout is reached
			if (difftime(time(NULL), t0) > _timeout){
				prev_frame.release();
				curr_frame.release();
				video.release();
				empty_frame_stock(in_btw_frm_stocked);
				delete vid_path_stack;
				DISPLAY(cout << "Timeout reached.\n");
				exit(EXIT_SUCCESS);
			}
		}

		// Cleaning memory
		prev_frame.release();
		curr_frame.release();
		video.release();

		// Relooping on the video to save in-between frames if necessary
		if (save_in_between_frames && !stock_in_between_frames){
			// Reload video
			video = VideoCapture(filepath);
			if(!video.isOpened()){
				cerr << "Couldn't open " << filepath << endl;
				continue;
			}
			// Parameters reinitialisation
			video >> curr_frame; // Getting second frame
			curr_frame_idx = video.get(CV_CAP_PROP_POS_FRAMES);
			
			unsigned int folder_id, frame_inf, frame_sup = 0;
			while(true){ // break is computed later (curr_frame_idx must be modified)
				// Getting next frame
				if (remove_identic_frames){
					prev_frame = curr_frame.clone();
				}
				curr_frame.release();
				video >> curr_frame;
				curr_frame_idx = video.get(CV_CAP_PROP_POS_FRAMES);
				if (curr_frame.empty()){
					break;
				}

				DISPLAY(cout << "Saving in-between frames... "
				<< format(" (%.2f %%) ", min(100., ((double) 100*curr_frame_idx)/_stop_at_frame))
				 << "\r");

				if (curr_frame_idx > frame_sup){
					if (in_btw_frm_indexes.empty()){
						break;
					}
					// Retrieving path to current animation
					folder_id = in_btw_frm_indexes.front();
					frame_path = out_dir+(string) "/"
							+to_string(folder_id)
							+(string) "/"
							+(string) BET_FRAMES_DIRNAME;
					in_btw_frm_indexes.pop();

					// Retrieving frames indexes
					frame_inf = in_btw_frm_indexes.front();
					in_btw_frm_indexes.pop();
					frame_sup = in_btw_frm_indexes.front();
					in_btw_frm_indexes.pop();

					// Creating "in-between" directory 
					dir_entry = fs::directory_entry(frame_path);
					if (!dir_entry.exists()){
						if (!fs::create_directory(dir_entry.path())){
							DISPLAY(cerr << "Failed to create directory " << dir_entry.path() << endl);
						}
					}
				}

				// Saving in-between frames
				if (frame_inf < curr_frame_idx && curr_frame_idx < frame_sup){
					if (!remove_identic_frames
						|| (remove_identic_frames && !are_identic_frames(prev_frame, curr_frame))){
						write_frame(curr_frame, frame_path, "frame_"+to_string(curr_frame_idx)+".jpg", false);
					}
				}

				// Cleaning memory and exiting program if timeout is reached
				if (difftime(time(NULL), t0) > _timeout){
					prev_frame.release();
					curr_frame.release();
					video.release();
					empty_frame_stock(in_btw_frm_stocked);
					delete vid_path_stack;
					DISPLAY(cout << "Timeout reached.\nLast complete in-between frame folder: " << folder_id-1 << endl);
					exit(EXIT_SUCCESS);
				}

			}
			curr_frame.release();
			video.release();
			DISPLAY(cout << endl)
		}
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
			delete[] this->diff; //Segfault here
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
			this->diff[pos[0]*prev->cols + pos[1]] = abs(sum(sum_diff)[0]) / (sub_prev.rows * sub_prev.cols); // Total difference ratio

			//TODO: Apply a log function to reduce the probability of having a 'nan'
			//TODO: Find a "fast" log approximation in c++
		}

		// Retrieve difference value
		// /!\ Call this after the forEach function, not before !!
		double getDiff(){
			double ret = 0;
			for (unsigned int i = 0; i < this->diff_len; i++){
				ret += this->diff[i];
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



#ifndef NOMAIN
/*======== PYTHON ==========*/

/* #include <boost/python.hpp>
BOOST_PYTHON_MODULE(exctractFrames){
    using namespace boost::python;
    def("exctractFrames", exctractFrames);
} */

/*======== MAIN ==========*/

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
	exctractFrames(argv[1], argv[2],
			30,    // skip_frames
			0,     // skip_seconds
			300,   // stop_at_frame
			1,     // display_interval
			20,    // min_mean_counted
			true,  // save_in_between_frames
			false, // stock_in_between_frames
			true,  // remove_identic_frames
			false, // compute_difference
			false,  // verbose
			1,     // diff_threshhold
			1,   // pic_save_proba
			0.05,  // file_proportion
			0,     // timeout
			NULL,  // first_frame_func
			NULL); // second_frame_func
    return 0;
}
#endif

