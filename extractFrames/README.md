# extractFrame function
=======================

This function writes frames from a video file into the output directory with various adjustable parameters. Also allow the possibility to save the in-between frames.

## Function prototype
---------------------
Here is the *exctractFrames* function prototype with all it's default values.

```c++
int exctractFrames(const char *in_path, const char *out_dir,
unsigned int skip_frames             = 1,
double       skip_seconds            = 0,
unsigned int start_at_frame          = 0,
unsigned int stop_at_frame           = 0,
unsigned int display_interval        = 1,
bool         save_in_between_frames  = true,
bool         stock_in_between_frames = true,
bool         remove_identic_frames   = false,
bool         compute_difference      = false,
unsigned int min_mean_counted        = 20,
double       diff_threshhold         = 0.20,
bool         verbose                 = true,
double       pic_save_proba          = 1,
double       file_proportion         = 1,
double       timeout                 = 0,     //In seconds
bool (*first_frame_func)(const Mat &)  = NULL,
bool (*second_frame_func)(const Mat &) = NULL,
bool (*compare_frame_func)(const Mat &, const Mat &) = NULL);
```

## Argument explaination
-------------------------

- **in_path** The path to a video file or a directory containing video files.
- **out_dir** The path to the output directory. If it doesn't exist, a new directory will be created.
- **skip_frames** The interval between each pair of frames that have to be compared and saved. For example, a *skip_frames* of 5 will compare and save frames 1-5, 6-11, 12-17...
- **skip_seconds** Same as skip_frames, but the unit is in seconds instead. If both *skip_frames* and *skip_seconds* are > 0, the priority is given to *skip_seconds*.
- **start_at_frame** The starting frame index.
- **stop_at_frame** The stoping frame index.
- **display_interval** The interval for information display. This is not by frame, but by loop. Each 'skip_frame' frames count for 1 loop. For example, having 'skip_frame' to 5 and *display_interval* to 3 will result in display at frames 1-5, 18-23...
- **save_in_between_frames** Whether or not in-between frames have to be saved.
- **stock_in_between_frames** Whether or not in-between frames have to be stocked in memory or saved later while relooping the video. It speeds up the function, but if too many frames are stored into the memory you might want to set this to *false*.
- **remove_identic_frames** Whether or not identic frames has to be removed or not. It is calculated for every successive frames so it slowers a bit the function.
- **compute_difference** Whether or not the difference between each pair of frames has to be calculated, in order to save only pictures that aren't very different. This slowers a lot the function.
- **min_mean_counted** The number of differences that have to be computed in order to have a meaningful mean.
- **diff_threshhold** The threshold idicating when frames are considered to too different (values are around 0~2).
- **verbose** Whether or not informations has to be displayed on the screen.
- **pic_save_proba** The probability to save a pair of frames or not (unselected pair won't compute difference or user-specified test functions)
- **file_proportion** The probability compute a found video or not (if too many videos are in the same directory).
- **timeout** A timeout in seconds. Note the if 'stock_in_between_frames' is set to 0, the last video's in-between frames might not be all saved.
- **first_frame_func** A user-specified function to test a property on the first frame. If this property if not verified, the pair of frames won't be saved.
- **second_frame_func** A user-specified function to test a property on the second frame. If this property if not verified, the pair of frames won't be saved.
- **compare_frame_func** A user-specified function to test a property on the pair of frames. If this property if not verified, the pair of frames won't be saved.


## Installation
----------------

Be sure to have *OpenCV* and **all its dependencies** installed on your computed. If compilation fails even with everything installed, set the Makefile's *INCLUDE* variable to the path to your OpenCV libraries.

In order to install the library, just download the repository. You can use the precompiled librairy, or you can compile it yourself from the source code and the Makefile. Compile with -DNOMAIN in order to make a librairy.



## How to use
--------------
### C++


### Python