import numpy
import pyfftw


def temporal_bandpass_filter(data, fps, freq_min=0.833, freq_max=1, axis=0):
    """Function that applies an ideal bandpass filter to the given signal.
       Created by: https://github.com/wzpan.
       Retrieved from: https://github.com/alessandro-gentilini/opencv_exercises-butterworth/issues/1"""
    # print("Applying bandpass between " + str(freq_min) + " and " + str(freq_max) + " Hz")
    # perform FFT on each frame
    fft = pyfftw.interfaces.numpy_fft.fft(data, axis=axis)
    # sampling frequencies, where the step d is 1/samplingRate
    frequencies = numpy.fft.fftfreq(data.shape[0], d=1.0 / fps)
    # find the indices of low cut-off frequency
    bound_low = (numpy.abs(frequencies - freq_min)).argmin()
    # find the indices of high cut-off frequency
    bound_high = (numpy.abs(frequencies - freq_max)).argmin()
    # band pass filtering
    fft[:bound_low] = 0
    fft[-bound_low:] = 0
    fft[bound_high:-bound_high] = 0
    # perform inverse FFT
    return numpy.real(pyfftw.interfaces.numpy_fft.ifft(fft, axis=0))
