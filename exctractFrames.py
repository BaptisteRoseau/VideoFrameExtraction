# Importing all necessary libraries 
import cv2 
import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
import threading as th
import time as tm

#TODO: be able to recreate the previous picture with the difference one
#TODO: make the script faster.... (or do it in Cpp ?!)

def differencePixel(previousPic, nextPic, outputPic, _i, _j, color, filterSize, lock=None):
	""" Computes locally the dfference between 'previousPic' and 'nextPic' for the pixel 'i', 'j' on the color 'color' according to the 'filterSize'.
	The result will be written into 'outputPic'. """
	tmp = (filterSize[0]//2, filterSize[1]//2)
	shape = previousPic.shape
	top, bot = max(_i - tmp[0], 0), min(_i + tmp[0] + 1, shape[0])
	left, right = max(_j - tmp[1], 0), min(_j + tmp[1] + 1, shape[1])
	s = 0
	for i in range(top, bot):
		for j in range(left, right):
			s += nextPic[i, j, color] - previousPic[i, j, color]
	s = s//((bot - top)*(right - left))
	outputPic.itemset((_i, _j, color), s)
	return s

def differenceLines(previousPic, nextPic, outputPic, diffArray, filterSize, threadID, i_min=0, i_max=None, lock=None):
	""" Warning: This function exits the program. This is used to exit current thread. """
	# Note: Use lock like this: "if lock is not None: lock.acquire()"
	shape = previousPic.shape
	assert shape == nextPic.shape and len(shape) == 3 and shape[2] == 3, "Pictures' shape must be the same: (_, _, 3)."
	assert i_min >= 0 and i_max <= shape[0]
	if i_max is None: i_max = shape[0]
	r, g, b = 0, 0, 0
	for i in range(i_min, i_max):
		for j in range(shape[1]):
			r += differencePixel(previousPic, nextPic, outputPic, i, j, 0, filterSize, lock)/3
			g += differencePixel(previousPic, nextPic, outputPic, i, j, 1, filterSize, lock)/3
			b += differencePixel(previousPic, nextPic, outputPic, i, j, 2, filterSize, lock)/3
	diffArray[threadID] = int(r+g+b)
	

def difference(previousPic, nextPic, nbThreads=8, filterSize=(5, 5)):
	""" Returns the picture difference between previousPic and nextPic, and it's value.
	INPUT:
	    - previousPic : The first picture
	    - nextPic     : The second picture (same size as previousPic)
	    - nbThreads   : The amount of threads used by this function (default 8)
	    - filterSize  : The size of the filter applied to determine the difference on one pixel (default (5, 5))
	OUTPUT:
	    - The tuple (differencePic : numpy.array of shape previousPic.shape, differenceValue : Integer) """
	shape = previousPic.shape
	assert shape == nextPic.shape and len(shape) == 3 and shape[2] == 3, "Pictures' shape must be the same: (_, _, 3)."
	step = shape[0] // nbThreads # Nb of lines computed par threads
	outputPic = np.zeros(previousPic.shape, np.int32)
	# Initializing threads
	diffArray   = [None] * nbThreads
	lock = th.Lock()
	threadArray = [
		th.Thread(
			target=differenceLines,
			args=(previousPic, nextPic, outputPic, diffArray, filterSize, i, i*step, (i+1)*step, lock)
		)
		for i in range(nbThreads)
	]
	# Running thread
	for thread in threadArray:
		thread.start()
	# Waiting threads and calculating difference
	s = 0
	for i in range(nbThreads):
		threadArray[i].join()
		s += diffArray[i]
	return outputPic, s

def difference_notThreaded(previousPic, nextPic, filterSize=(5, 5)):
	shape = previousPic.shape
	assert shape == nextPic.shape and len(shape) == 3 and shape[2] == 3, "Pictures' shape must be the same and in format (_, _, 3)."
	outputPic = np.zeros(previousPic.shape, np.int32)
	tmp = (filterSize[0]//2, filterSize[1]//2)
	shape = previousPic.shape
	rgb = np.zeros(3, np.uint32)
	for i in np.arange(shape[0]):
		for j in range(shape[1]):
			top, bot    = max(i - tmp[0], 0), min(i + tmp[0] + 1, shape[0])
			left, right = max(j - tmp[1], 0), min(j + tmp[1] + 1, shape[1])
			nextSubArray = nextPic[top:bot, left:right, :]
			prevSubArray = previousPic[top:bot, left:right, :]
			rgb += np.array([
				np.sum(nextSubArray[:, :, 0]) - np.sum(prevSubArray[:, :, 0]),
				np.sum(nextSubArray[:, :, 1]) - np.sum(prevSubArray[:, :, 1]),
				np.sum(nextSubArray[:, :, 2]) - np.sum(prevSubArray[:, :, 2]),
				])//(3*(bot - top)*(right - left))
	return outputPic, np.sum(rgb)

def process(video_path, output_dir, write=False, threshold=None, skip_frames=None, skip_seconds=None, nb_frames_max=np.Inf, timeout=np.Inf):
	""" Main process """ #TODO: Ecrire cette doc
	assert skip_frames is None or skip_seconds is None, "Cannot skip by seconds and frames. Please choose one."

	# Read the video from specified path 
	video = cv2.VideoCapture(video_path) 
	fps = video.get(cv2.CAP_PROP_FPS)

	# Set the frameSkip to skip some frames
	if skip_seconds is not None:
		frameSkip = int(fps*skip_seconds)
	elif skip_frames is not None:
		frameSkip = skip_frames
	else:
		frameSkip = 1

	# Creating the output directory
	try: 
		if not os.path.exists(output_dir): 
			os.makedirs(output_dir) 
	# if not created then raise error 
	except OSError: 
		print ('Error: Creating directory of data: '+output_dir) 

	# Initialisation
	diff_arr = [] # Arrays containing the differenceSum
	mean_arr = [] # The evolution of the mean
	ret,currentFrame = video.read()
	currentFrameIdx = 1
	mean = 0
	_write = write # Loop local variable for writing
	t0 = tm.time()

	# Retrieving data information
	while(currentFrameIdx < nb_frames_max and tm.time() - t0 < timeout):
		previousFrame = currentFrame

		# Skiping 4 frames
		for _ in range(frameSkip):
			ret,currentFrame = video.read() 
			currentFrameIdx += 1

		if ret:
			# Calculating the difference between the 2 pictures
			differenceFrame, diff = difference_notThreaded(previousFrame, currentFrame)
			#differenceFrame, diff = difference(previousFrame, currentFrame)
			if mean == 0: mean += diff
			
			#Saving the difference into diff_arr
			diff_arr.append(diff)
			mean_arr.append(mean)
			mean += (diff - mean)/len(diff_arr)

			# Detect if the two images are very differents (according to the threshold)
			if not threshold is None and diff >= threshold: _write = True
			if currentFrameIdx > 5*frameSkip and mean != 0 and abs((diff - mean))/mean > 1.5: _write = True #TODO: Ajuster le 1.5 qui est ici

			# writing the extracted images 
			if _write:
				namePrev = os.path.normpath('./'+output_dir+'/frame' + str(currentFrameIdx) + '_prev.jpg')
				nameCurr = os.path.normpath('./'+output_dir+'/frame' + str(currentFrameIdx) + '_curr.jpg')
				nameDiff = os.path.normpath('./'+output_dir+'/frame' + str(currentFrameIdx - 1) + '_diff.jpg')
				nameRedo = os.path.normpath('./'+output_dir+'/frame' + str(currentFrameIdx - 1) + '_Redo.jpg')
				print ('Creating ' + nameCurr)
				cv2.imwrite(nameDiff, abs(differenceFrame)) 
				cv2.imwrite(nameCurr, currentFrame) 
				cv2.imwrite(namePrev, previousFrame)
				cv2.imwrite(nameRedo, previousFrame + differenceFrame) 

			if mean != 0: print(currentFrameIdx, '\t', mean, '\t', diff, '\t', abs((diff - mean)/mean))
			# increasing frame counter
			currentFrameIdx += 1
			_write = write
		else: 
			break

	# Release all space and windows once done 
	video.release() 
	cv2.destroyAllWindows()

	return diff_arr, mean_arr, mean, skip_frames

#=============================================================================================================================================
#                      MAIN
#=============================================================================================================================================

# Retrieving input video
if len(sys.argv) > 1:
	PATH = os.path.normpath(os.path.realpath(os.path.join(os.path.dirname(__file__), sys.argv[1])))
else:
	PATH = "C:\\MangaAnimation\\RandomScriptTests\\DatabaseGeneration\\AccelWorld1.mp4"

# Main process
arr, marr, mean, skpFrames = process(PATH,
		output_dir='tesuto',
		write=True,
		threshold=None,
		skip_seconds=None,
		skip_frames=24,
		nb_frames_max=500,
		timeout=300
)

plt.plot(skpFrames*np.arange(len(arr)), arr, label="Differences")
plt.plot(skpFrames*np.arange(len(marr)), marr, label="Mean Evolution")
plt.title("Picture difference between each frames")
plt.xlabel("Frame")
plt.ylabel("Difference")
plt.grid()
plt.show()
print("Exited Successully.\nMean:", mean)

