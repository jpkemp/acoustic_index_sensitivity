import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from os.path import basename
from scipy import signal, fftpack
import numpy as np
import matplotlib.pyplot as plt

def pcm2float(sig, dtype='float64'):
    """Convert PCM signal to floating point with a range from -1 to 1.

    Use dtype='float32' for single precision.

    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.

    Returns
    -------
    numpy.ndarray
        Normalized floating point data.

    See Also
    --------
    float2pcm, dtype

    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max

#------------------------------------------------------------------------
def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.

    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.

    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html

    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.

    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        `dtype`.

    See Also
    --------
    pcm2float, dtype

    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

class AudioFile(object):

    def __init__(self, file_path, verbose=False):

        default_channel = 0  # Used channel if the audio file contains more than one channel
        #default_channel = 1

        if verbose:
            print('Read the audio file:', file_path)
        try:
            sr, sig = wavread(file_path)
        except IOError:
            print("Error: can\'t read the audio file:", file_path)
        else:
            if verbose:
                print('\tSuccessful read of the audio file:', file_path)
            if len(sig.shape)>1:
                if verbose:
                    print('\tThe audio file contains more than one channel. Only the channel', default_channel, 'will be used.')
                sig=sig[:, default_channel] # Assign default channel
            self.sr = sr
            self.sig_int = sig
            self.sig_float = pcm2float(sig,dtype='float64')
            self.niquist = sr/2
            self.file_path = file_path
            self.file_name = basename(file_path)
            self.filtered = False
            self.duration = len(sig)/float(sr)
            self.indices = dict()  # empty dictionary of Index

    def process_filtering(self, sig_float_filtered, write=False, output_file_name = None):
        self.filtered = True
        self.sig_int = float2pcm(sig_float_filtered)
        self.sig_float = sig_float_filtered
        if write:
            wavwrite(output_file_name, file.sr, file.sig_int)

def compute_spectrogram(file, windowLength=512, windowHop= 256, scale_audio=True, square=True, windowType='hanning', centered=False, normalized = False ):
    """
    Compute a spectrogram of an audio signal.
    Return a list of list of values as the spectrogram, and a list of frequencies.

    Keyword arguments:
    file -- the real part (default 0.0)

    Parameters:
    file: an instance of the AudioFile class.
    windowLength: length of the fft window (in samples)
    windowHop: hop size of the fft window (in samples)
    scale_audio: if set as True, the signal samples are scale between -1 and 1 (as the audio convention). If false the signal samples remains Integers (as output from scipy.io.wavfile)
    square: if set as True, the spectrogram is computed as the square of the magnitude of the fft. If not, it is the magnitude of the fft.
    hamming: if set as True, the spectrogram use a correlation with a hamming window.
    centered: if set as true, each resulting fft is centered on the corresponding sliding window
    normalized: if set as true, divide all values by the maximum value
    """

    if scale_audio:
        sig = file.sig_float # use signal with float between -1 and 1
    else:
        sig = file.sig_int # use signal with integers

    W = signal.get_window(windowType, windowLength, fftbins=False)
    halfWindowLength = int(windowLength/2)

    if centered:
        time_shift = int(windowLength/2)
        times = range(time_shift, len(sig)+1-time_shift, windowHop) # centered
        frames = [sig[i-time_shift:i+time_shift]*W for i in times] # centered frames
    else:
        times = range(0, len(sig)-windowLength+1, windowHop)
        frames = [sig[i:i+windowLength]*W for i in times]

    if square:
        spectro =  [abs(np.fft.rfft(frame, windowLength))[0:halfWindowLength]**2 for frame in frames]
    else:
        spectro =  [abs(np.fft.rfft(frame, windowLength))[0:halfWindowLength] for frame in frames]

    spectro=np.transpose(spectro) # set the spectro in a friendly way

    if normalized:
        spectro = spectro/np.max(spectro) # set the maximum value to 1 y

    frequencies = [e * file.niquist / float(windowLength / 2) for e in range(halfWindowLength)] # vector of frequency<-bin in the spectrogram
    return spectro, frequencies





#-----------------------------------------------------------------------------
def compute_ACI(spectro,j_bin):
    """
    Compute the Acoustic Complexity Index from the spectrogram of an audio signal.

    Reference: Pieretti N, Farina A, Morri FD (2011) A new methodology to infer the singing activity of an avian community: the Acoustic Complexity Index (ACI). Ecological Indicators, 11, 868-873.

    Ported from the soundecology R package.

    spectro: the spectrogram of the audio signal
    j_bin: temporal size of the frame (in samples)


    """

    #times = range(0, spectro.shape[1], j_bin) # relevant time indices
    times = range(0, spectro.shape[1]-10, j_bin) # alternative time indices to follow the R code

    jspecs = [np.array(spectro[:,i:i+j_bin]) for i in times]  # sub-spectros of temporal size j

    aci = [sum((np.sum(abs(np.diff(jspec)), axis=1) / np.sum(jspec, axis=1))) for jspec in jspecs] 	# list of ACI values on each jspecs
    main_value = sum(aci)
    temporal_values = aci

    return main_value, temporal_values # return main (global) value, temporal values
