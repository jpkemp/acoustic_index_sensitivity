import pickle
from abc import ABC, abstractmethod
from datetime import datetime as dt, timedelta as td
from math import log
from pathlib import Path
from multiprocessing import Pool
from typing import Callable
import pandas as pd
import numpy as np
from maad.features import acoustic_complexity_index, bioacoustics_index, acoustic_diversity_index, acoustic_eveness_index
from maad.sound import spectrogram
from scipy.signal import butter, sosfilt
from tqdm import tqdm
from tools.r_link import Rlink
from tools.sound_processor import SoundProcessor as sp
from study_settings.common import CommonSettings

class IndexSensitivity(ABC):
    '''Functions for loading sound files, calulating acoustic indices, and modelling differences at various parameters. These functions have been designed with this study in mind and some may not generalise well'''
    def __init__(self, 
                 settings:CommonSettings,
                 window_sizes:list, 
                 test_bands:list, 
                 indices_of_interest:list) -> None:
        '''settings: dataclass of settings for the study
           window_size: FFT window sizes to test
           test_bands: list of tuples in the format (min_freq, max_freq) to be either band-pass filtered or extracted from the spectrogram
           indices_of_interest: list of indices to test. Available options are "ACI", "ADI", "AEI", and "BIO"
           check_file: function to extract site and timestamp information from file and folder names
           series_definition: function to create a pandas series for the dataframe
        '''
        self.settings = settings
        self.fft_windows = window_sizes
        self.test_bands = test_bands
        self.indices_of_interest = indices_of_interest

    @classmethod
    @abstractmethod
    def check_file(cls, folder:Path, file:Path):
        '''retrieve site and timestamp information from the folder and filenames'''

    @classmethod
    @abstractmethod
    def series_definition(cls, index, band_name, filtered, stamp, site, parameter, func, Sxx, fn, truncation):
        '''series creation for rows of the dataframe, including column names, which vary slightly between the Carara and Big Vicky experiments'''

    @classmethod
    @abstractmethod
    def create_dataframe(cls, serieses:list, input_path:str=None, output_path:str=None):
        '''concatenates a list of series to a dataframe, and converts columns to the correct types
        
        input_path: if given, loads an existing dataframe before setting the types, instead of concatenating a series
        output_path: if given, saves the dataframe to file'''


    @classmethod
    def samples_to_s(cls, sample_rate:int, samples:int) -> float:
        '''returns samples / sample_rate'''
        return samples / sample_rate

    @classmethod
    def s_to_samples(cls,sample_rate:int, time:float) -> float:
        '''returns sample_rate * time'''
        return sample_rate * time

    @classmethod
    def get_index_func(cls, idx:str) -> Callable:
        '''get an acoustic index function.

        idx: Available options are "ACI", "ADI", "AEI", and "BIO"

        returns: Callable'''
        if idx == "ACI": return lambda x, y: acoustic_complexity_index(x)[2]
        if idx == "ADI": return lambda x, y: acoustic_diversity_index(x, y, fmin=min(y), fmax=max(y), bin_step=100)
        if idx == "AEI": return lambda x, y: acoustic_eveness_index(x, y, fmin=min(y), fmax=max(y), bin_step=100)
        if idx == "BIO": return lambda x, y: bioacoustics_index(x, y, (min(y), max(y)))

        raise ValueError(f"No applicable index function for {idx}")

    @classmethod
    def get_adi_truncation(cls, y, bin_step=100):
        lower_freq = min(y)
        upper_freq = max(y)
        return log(np.floor((upper_freq-lower_freq)/bin_step))

    @classmethod
    def get_rounded_timestamp(cls, timestamp:str, nearest_minute:int, fmt:str) -> dt:
        '''Round timestamp to the nearest minute
        
        timestamp: the timestamp
        nearest_minute: minute to round to (<60)
        fmt: format sting for the timestamp
        
        returns: datetime object with the rounded timestamp'''
        stamp = dt.strptime(timestamp, fmt)
        discard = td(minutes=stamp.minute % nearest_minute,
                        seconds=stamp.second,
                        microseconds=stamp.microsecond)

        stamp -= discard
        if discard >= td(minutes=nearest_minute // 2, seconds=30 * (nearest_minute % 2)):
            stamp += td(minutes=nearest_minute)

        return stamp

    @classmethod
    def pickle_data(cls, data:object, filepath: str) -> None:
        '''pickle an object to file
        
        data: the object to pickle
        filepath: path to save to'''
        filepath = str(filepath)
        if filepath[-4:] != ".pkl":
            filepath = filepath + ".pkl"

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def unpickle_data(cls, filepath:str) -> object:
        '''load a pickle object from file
        
        filepath: the file to load
        
        returns: object'''
        filepath = str(filepath)
        if filepath[-4:] != ".pkl":
            filepath = filepath + ".pkl"

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        return data

    def remove_below_threshold(self, Sxx, fn):
        '''remove frequencies from a spectrogram below a threshold. The threshold is defined in the settings supplied on instance creation. 
        
        Sxx: the spectrogram
        fn: the frequencies matching the spectrogram'''
        n_below_threshold = len(fn[fn < self.settings.frequency_threshold])

        return Sxx[n_below_threshold:, :], fn[n_below_threshold:]

    def filter_frequencies_from_sound(self, sound, band, window):
        fltr = butter(5, band, 'bandpass', output='sos', fs=sound.fs)
        fltrd_sig = sosfilt(fltr, sound.signal)
        fltr_Sxx, fltr_tn, fltr_fn, fltr_ext = spectrogram(fltrd_sig, sound.fs, window=self.settings.window, nperseg=window, noverlap=0)
        if self.settings.frequency_threshold:
            fltr_Sxx, fltr_fn = self.remove_below_threshold(fltr_Sxx, fltr_fn)

        fn = fltr_fn
        Sxx = fltr_Sxx

        return Sxx, fn

    @classmethod
    def extract_frequencies_from_sxx(cls, sxx, fn, band):
        above = len(fn) - len(fn[fn > band[1]])
        below = len(fn[fn < band[0]])
        post_fn = fn[below:above]
        post_sxx = sxx[below:above]

        return post_sxx, post_fn

    def process_sound_file(self, filepath:Path, folder:Path):
        ''' Calculates index values for one sound file
            file: the file name
            folder: location of the file

            returns: a list of pandas series containing the index values and other information for all indices of interest
        '''
        file_info = self.check_file(folder, filepath)
        if not file_info:
            return None

        site, stamp = file_info
        wave = sp.open_wav(filepath, trim_start=self.settings.trim_file_start, channel=1)
        ret = []
        for parameter in self.fft_windows:
            pre_Sxx, tn, pre_fn, ext = spectrogram(wave.signal, wave.fs, window=self.settings.window, nperseg=parameter, noverlap=0)
            if self.settings.frequency_threshold:
                pre_Sxx, pre_fn = self.remove_below_threshold(pre_Sxx, pre_fn)

            for band_name, freq, filtered in self.test_bands:
                if filtered:
                    Sxx, fn = self.filter_frequencies_from_sound(wave, freq, parameter)
                else:
                    Sxx, fn = self.extract_frequencies_from_sxx(pre_Sxx, pre_fn, freq)

                for index in self.indices_of_interest:
                    func = self.get_index_func(index)
                    truncation = self.get_adi_truncation(fn) if index == "ADI" else 0
                    p = self.series_definition(index, band_name, filtered, stamp, site, parameter, func, Sxx, fn, truncation)
                    ret.append(p)

        return ret

    def build_model(self, r_link:Rlink, marine:bool, text_options:list, df:pd.DataFrame, factors:list, truncation=0):
        ''' Build a GLMM model for the index values.
        r_link: an instance of Rlink 
        marine: whether the data is from Big Vicky or Carara
        text_options: tuple of descriptions of index, filtered, band_name, cross_effect 
        df: the index value data
        factors: which columns to treat as factors in the model

        returns: the model and the conditional effects
        '''
        index, flt, band_name, cross_effect = text_options
        r_df = r_link.convert_to_rdf(df)
        for factor in factors:
            r_link.change_col_to_factor(r_df, factor)

        print(f"{band_name} {flt}using {index}")
        path = f"output/{cross_effect} x Window Conditional Effects for {index.upper()} over {flt}{band_name.lower()} frequencies"
        model, effects = r_link.r_src.find_effects(r_df, index, str(path), marine=marine, 
                                        iter=self.settings.iterations, warmup=self.settings.warmup, upper_bound=float(truncation))
        print(model)
        warnings = r_link.r_src.get_warnings()
        if warnings != r_link.null_value:
            print(warnings)

        return model, effects[-1]

    def get_index_values(self, input_file:str=None, output_file:str=None):
        ''' Calculate all index values for all relevant files, based on settings supplied at instance creation

        input_file: if given, loads data from file instead of calculating it
        output_file: if given, and if input_file is not given, saves the data to file

        returns: list of series objects for concatenation and further processing
        '''
        n_processes = self.settings.n_processes
        if input_file:
            return self.unpickle_data(input_file=None)

        serieses:list[pd.DataFrame] = []
        for folder in Path(self.settings.data_location).iterdir():
            files  = list(folder.glob("*.wav"))
            with Pool(n_processes) as pool, tqdm(total=len(files), leave=False) as pbar:
                ret = [pool.apply_async(self.process_sound_file, args=(i, folder), callback=lambda _:pbar.update(1)) for i in files]
                res = [r.get() for r in ret]
                for r in res:
                    if r is not None:
                        for s in r:
                            serieses.append(s)

        if output_file:
            self.pickle_data(serieses, output_file)

        return serieses
