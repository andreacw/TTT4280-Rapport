from lab4 import *

# read and extract data from file
# path, fps = read_video()
fps = 40
data = read_file('datafiler/reflektans3')

# defining color channels
colors = ["red", "green", "blue"]

# initialize empty arrays
corrs = []
lags = []
peaks = []
heights = []
pulses = []


# Butterworth
T = 30.0        # Sample Period
cutoff = (MINPULSE/60)      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fps  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fps) # total number of samples


# plots detrended data
plt.figure(figsize=(8,6))
plt.title("Detrended data in the time domain", fontsize=12)
for i in range(3):
    plt.plot(np.linspace(0, 30, len(data)), data[:, i], linestyle='dashed', color=colors[i], label=colors[i])
    plt.xlabel('time (s)')
for i in range(3):
    data[:,i] = butter_highpass_filter(data[:,i], cutoff, fps, order)
    plt.plot(np.linspace(0, 30, len(data)), data[:, i], linestyle='solid', color=colors[i], label=colors[i])
    plt.xlabel('time (s)')
plt.legend()
plt.show()


# runs the functions defined in utility functions for each color channel
for i in range(3):
    lag, corr, _, _ = plt.xcorr(data[:, i], data[:, i], normed=False, usevlines=False, maxlags=WINDOW//2)
    plt.close()
    peak, height = peak_finder(corr)
    puls = pulse_finder(peak, fps)
    corrs.append(corr)
    lags.append(lag)
    peaks.append(peak)
    heights.append(height)
    pulses.append(puls)
    print("Pulse for ", colors[i], " channel: \t", puls)

# creates plots for crosscorrelation
plt.figure(figsize=(6,8))
plt.suptitle("Cross correlation, filtered data and peaks found", fontsize=12)
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.title(colors[i].capitalize(), fontsize=10)
    plt.plot(lags[i], corrs[i], linestyle='solid', color=colors[i])
    plt.plot(lags[i], heights[i], linestyle='dashed')
    plt.plot((peaks[i]-(WINDOW/2)), corrs[i][peaks[i]], "X")

plt.subplots_adjust(hspace=0.3)
plt.show()
