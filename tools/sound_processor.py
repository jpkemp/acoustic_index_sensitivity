import math
import numpy as np
import wave
from scipy.io import wavfile
from scipy import signal as sig

WAV_MAX = 32767
WAV_MIN = -32768
WAV_ABS_MAX = 32768
WAV_RANGE = 2**16

class Sound:
    '''Container for sound information.
       Parameters:
       signal: the sound signal (vector-like)
       time: vector-like with timestamps corresponding to the samples
       fs: sample frequency
       duration: length of the signal in seconds
    '''
    def __init__(self, signal, time, fs):
        self.signal = signal
        self.fs = fs
        self.time = time
        self.duration = len(signal) / self.fs

    def rescale_signal(self):
        '''Re-scales the signal between 0 and 1, if the signal starts between -1 and 1'''
        self.signal = (self.signal + 1) / 2

class SoundProcessor:
    '''Functions for processing sound signals'''
    @classmethod
    def normalise_sound(cls, data, data_format):
        '''Scales data to range [-1, 1] based on the wav format'''
        def temp(x):
            if not x:
                return 0.0
            if x > 0:
                return float(x) /np.iinfo(data_format).max
            
            return float(x) / abs(np.iinfo(data_format).min)

        conv = np.vectorize(temp)
        return conv(data)

    @classmethod
    def open_wav(cls, input_file:str, channel:int=None, trim_start:int=0, length:int=-1, soundtrap:int=None, normalise:bool=True)-> Sound:
        ''' Open a wave file
            input_file: file to open

            Optional arguments:
            channel: channel to use in a multi-channel file. Defaults to None
            trim_start: cut the start of a file (in seconds)
            length: length of the signal to keep (in seconds)
            soundtrap: soundtrap ID for calibration

            Returns a Sound object
        '''
        wave_format = {1: np.uint8, 2: np.int16, 4:np.int32}
        with wave.open(str(input_file), 'rb') as file:
            nch = file.getnchannels()
            sampwidth = file.getsampwidth()
            frames = file.readframes(-1)
            nframes = file.getnframes()
            fs = file.getframerate()
            if sampwidth == 3:
                a = np.ndarray((nframes * nch * sampwidth), dtype=np.uint8, buffer=frames)
                b = np.empty((nframes, nch, 4), dtype=np.uint8)
                b[:, :, :sampwidth] = a.reshape(-1, nch, sampwidth)
                b[:, :, :sampwidth] = (b[:, :, sampwidth - 1:sampwidth] >> 7) * 255
                a = b.view('<i4').reshape(b.shape[:-1])
            else:
                a = np.ndarray((nframes * nch,), dtype=wave_format[sampwidth], buffer=frames)

            if normalise:
                a = cls.normalise_sound(a, wave_format[sampwidth])

        if nch > 1:
            if channel is None:
                print("Warning: no channel set for a multi-channel file. Using channel 1")
                channel = 1
            channels = [ [] for _ in range(nch) ]
            for index, value in enumerate(a):
                channels[index % nch].append(value)

            signal = np.array(channels[channel - 1])
            # signal = int_data[:, channel - 1]
        else:
            if channel is not None and channel != 1:
                print("Warning: invalid channel set for single-channel file. Using channel 1")

            signal = a

        trim_start = fs * trim_start
        if length > 0:
            length = trim_start + (fs * length)
            signal = signal[trim_start:length]
        else:
            signal = signal[trim_start:]

        if soundtrap:
            signal = cls.sountrap_conversion(signal, soundtrap)

        time = np.arange(0, signal.size / fs, 1/fs)
        if len(time) == len(signal) + 1:
            time = time[:-1]

        return Sound(signal, time, fs)

    @classmethod
    def get_wav_length(cls, filename):
        with wave.open(str(filename), 'rb') as f:
            return f.getnframes()

    @classmethod
    def get_sample(cls, sound:Sound, start:float, end:float) -> Sound:
        '''get a subsample from a sound, with a start and end time
        
        sound: a Sound object
        start: the start time
        end: the end time
        
        returns: a new Sound object with the subsample'''
        start_samp = int(start * sound.fs)
        end_samp = int(end * sound.fs)

        samp = sound.signal[start_samp:end_samp]
        time = np.arange(0, samp.size / sound.fs, 1/sound.fs)
        if len(time) == len(samp) + 1:
            time = time[:-1]

        sound = Sound(samp, time, sound.fs)

        return sound 

    @classmethod
    def plot_signal(cls, sound:Sound, output_path:str, plotter, title=None)->None:
        '''Wrapper for plotting sound signals

           sound: a Sound object
           output_path: path to save the plot

           Optional arguments:
           title: Title for the plot. Defaults to None

           Returns None
           '''
        units = ['s', 'dB']
        plotter.plot_signal(sound.time, sound.signal, units, output_path, title=title)

    @classmethod
    def create_sine(cls, f:int, t:float, fs:int, amp=WAV_MAX, output_path:str=None)->Sound:
        '''Create a sine wave in wav format.
           f: fundamental frequency
           t: duration
           fs: sample frequency

           Optional arguments:
           amp: max amplitude. Defaults to 32767, for .wav
           output_path: writes to .wav if not None. Defaults to None
           data_format: format for the wav file. Defaults to 16 bit

           Returns a Sound object
        '''
        rad_f = 2 * np.pi * f
        samples = np.arange(fs * t) / fs
        sgnl = np.sin(rad_f * samples) * amp
        if output_path:
            amp = max(max(sgnl), abs(min(sgnl)))
            converted_sig = None
            for data_format in [np.int16, np.int32]:
                if amp <= np.iinfo(data_format).max:
                    converted_sig = np.array([x for x in sgnl], dtype=data_format)
                    wavfile.write(output_path, fs, converted_sig)
                    break
            else:
                raise ValueError(f"Amplitude {amp} is not appropriate for uint8, int16, or int32")

        time_vec = np.arange(0, t, 1/fs)
        sound = Sound(sgnl, time_vec, fs)

        return sound

    @classmethod
    def create_spectrogram(cls, signal, fs, samples=512, window='hann', overlap=0, mode='psd', plotter=None, filename=None, title=None)->tuple:
        '''Create a spectrogram from a sound signal

           sound: a Sound object

           Optional arguments:
           samples: FFT window length
           window: FFT window type
           overlap: Overlap between FFT windows in samples
           output_path: Output path for the plot. Defaults to None (no file saved)
           title: Title for the plot. Defaults to None

           Returns (f, t, sxx) where f is a vector-like of the frequencies, t is a vector-like of the times, and sxx is the spectrogram
        '''
        window = sig.get_window(window, samples)
        f, t, sxx = sig.spectrogram(signal, fs, window=window, noverlap=overlap, mode=mode, scaling='density', nperseg=samples, nfft=samples, axis=-1)
        # sxx *= 10e16
        if filename and plotter:
          plotter.plot_spectrogram(f, t, sxx, filename, title)

        return f, t, sxx

    @classmethod
    def calculate_aci(cls, sound:Sound, timestep:float, spectro:tuple=None)->float:
        '''Calculate the ACI of a signal
        sound: a Sound object
        timestep: the window to calculate ACI over (in seconds, if the fs of the sound is in Hz)

        Optional arguments:
        spectro: (f, t, sxx) from the create_spectrogram function. Defaults to None (create_spectrogram is run)

        Returns total ACI for the signal
        '''
        if spectro is None:
            f, t, sxx = cls.create_spectrogram(sound)
        else:
            f, t, sxx = spectro

        n_samples = int(math.ceil(timestep / (t[1] - t[0])))
        js = [sxx[:,idx:idx + n_samples] for idx in range(0, sxx.shape[1], n_samples)]

        aci_per_freq = []
        for j in js:
            den = np.sum(j, axis=1)
            D = np.sum(np.abs(np.diff(j)), axis=1)
            aci_per_j = sum(D / den)
            aci_per_freq.append(aci_per_j)

        return sum(aci_per_freq)

    @classmethod
    def write_wav(file, sound:Sound):
        with wave.open(file, 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sound.fs)
            f.setnframes(len(sound.signal))
            f.writeframes(sound.signal)
