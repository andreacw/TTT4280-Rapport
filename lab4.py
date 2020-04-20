import sys
# sys.path.append('/usr/local/lib/python2.7/site-packages') # unnecessary if you don't have trouble with the cv2 package
import cv2 
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

WINDOW = 250
MAXPULSE = 230
MINPULSE = 40
fps = 40

# todo: lese og skrive om matlabkode til python
# todo: plotte signaler før og etter i fft

# read_video() is a function based on the code found in "read_video_and_extract_roi.py"
# it has no input and returns a path to exctracted data as a string and frames per second
def read_video():
    #CLI options
    if len(sys.argv) < 3:
        print("Select smaller ROI of a video file, and save the mean of each image channel to file, one column per color channel (R, G, B), each row corresponding to a video frame number.")
        print("")
        print("Usage:\npython " + sys.argv[0] + " [path to input video file] [path to output data file]")
        exit()
    filename = sys.argv[1]
    output_filename = sys.argv[2]

    #read video file
    cap = cv2.VideoCapture(filename, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Could not open input file. Wrong filename, or your OpenCV package might not be built with FFMPEG support. See docstring of this Python script.")
        exit()

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    mean_signal = np.zeros((num_frames, 3))

    #loop through the video
    count = 0
    while cap.isOpened():
        ret, frame = cap.read() #'frame' is a normal numpy array of dimensions [height, width, 3], in order BGR
        if not ret:
            break

        #display window for selection of ROI
        if count == 0:
            window_text = 'Select ROI by dragging the mouse, and press SPACE or ENTER once satisfied.'
            ROI = cv2.selectROI(window_text, frame) #ROI contains: [x, y, w, h] for selected rectangle
            cv2.destroyWindow(window_text)
            print("Looping through video.")

        #calculate mean
        cropped_frame = frame[ROI[1]:ROI[1] + ROI[3], ROI[0]:ROI[0] + ROI[2], :]
        mean_signal[count, :] = np.mean(cropped_frame, axis=(0,1))
        count = count + 1

    cap.release()

    #save to file in order R, G, B.
    np.savetxt(output_filename, np.flip(mean_signal, 1))
    print("Data saved to '" + output_filename + "', fps = " + str(fps) + " frames/second")

    return output_filename, fps


# read_file(path) takes in the path of a data file, and returns it as type float64
# it returns data as 3 by 3 np.array, and uses signal.detrend to remove DC-components
def read_file(path):
    data = []
    f = open(path, "r")
    for line in f:
        line = line.split(" ")
        for val in line:
            val = val.strip(" ")
            val = val.strip("\n")
        data.append(line)
    f.close()
    data = np.array(data)
    data = data.astype('float64')
    data = signal.detrend(data[100:], axis=0)  # removes DC component
    return data



def butter_highpass_filter(data, cutoff, fps, order):
    normal_cutoff = cutoff / (0.5 * fps)
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

# savgol_filter implements a 4. order savgol filter
def savgol_filter(data, width): 
    return signal.savgol_filter(data, width-1, 2)

# takes cross corrolation and returns array of peaks and filtered heights
# also removes peaks that are so close they produce a pulse larger than MAXPULSE
def peak_finder(corr):
    height = savgol_filter(corr, WINDOW)
    peaks, _ = signal.find_peaks(corr, height=height, distance=(40*60)//MAXPULSE)   # the distance arg sets a minimum distance between peaks
   # peaks, _ = signal.find_peaks(corr, distance=(40*60)//MAXPULSE)   # the distance arg sets a minimum distance between peaks

    return peaks, height

# pulse_finder(peaks) takes in an array of peaks and calculates pulse
# it also removes values outside
# todo: produces error if all values are filtered out, errorhandling is unsucessful, runs fine with error
def pulse_finder(peaks, fps):
    pulse = fps*60/np.diff(peaks)
    try:
        pulse = pulse[(abs(pulse - np.median(pulse)) < abs((2 * np.std(pulse)+2))) & (pulse > MINPULSE)] # MAXPULSE is checked in the peak_finder function         
    except RuntimeWarning as e:
        print("Invalid data for one or more channels", e)
        return None
    return np.mean(pulse)
