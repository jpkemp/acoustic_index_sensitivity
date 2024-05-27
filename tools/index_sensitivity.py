import pickle
from datetime import datetime as dt, timedelta as td
from pathlib import Path
from multiprocessing import Pool
from typing import Callable
import pandas as pd
from maad.features import acoustic_complexity_index, bioacoustics_index, acoustic_diversity_index, acoustic_eveness_index
from maad.sound import spectrogram
from scipy.signal import butter, sosfilt
from tqdm import tqdm
from tools.sound_processor import SoundProcessor as sp
from datetime import datetime as dt, timedelta as td
import pandas as pd
from maad.features import acoustic_complexity_index, bioacoustics_index, acoustic_diversity_index, acoustic_eveness_index
from maad.sound import spectrogram
from scipy.signal import butter, sosfilt

class IndexSensitivity:
    def __init__(self, 
                 settings, 
                 window_sizes, 
                 test_bands:list, 
                 indices_of_interest:list, 
                 check_file:Callable, 
                 series_definition:Callable) -> None:
        self.settings = settings
        self.fft_windows = window_sizes
        self.test_bands = test_bands
        self.valid_file = check_file
        self.indices_of_interest = indices_of_interest
        self.series_definition = series_definition

    @classmethod
    def samples_to_s(cls, sample_rate, samples):
        return samples / sample_rate

    @classmethod
    def s_to_samples(cls,sample_rate, time):
        return sample_rate * time

    @classmethod
    def get_index_func(cls, idx):
        if idx == "ACI": return lambda x, y: acoustic_complexity_index(x)[2]
        if idx == "ADI": return acoustic_diversity_index
        if idx == "AEI": return acoustic_eveness_index
        if idx == "BIO": return bioacoustics_index

        raise ValueError(f"No applicable index function for {idx}")

    @classmethod
    def get_rounded_timestamp(cls, timestamp, nearest_minute, fmt):
        stamp = dt.strptime(timestamp, fmt)
        discard = td(minutes=stamp.minute % nearest_minute,
                        seconds=stamp.second,
                        microseconds=stamp.microsecond)

        stamp -= discard
        if discard >= td(minutes=nearest_minute // 2, seconds=30 * nearest_minute % 2):
            stamp += td(minutes=nearest_minute)

        return stamp

    @classmethod
    def pickle_data(cls, data, filepath) -> None:
        filepath = str(filepath)
        if filepath[-4:] != ".pkl":
            filepath = filepath + ".pkl"

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def unpickle_data(cls, filepath):
        filepath = str(filepath)
        if filepath[-4:] != ".pkl":
            filepath = filepath + ".pkl"

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        return data

    def remove_below_threshold(self, Sxx, fn):
        n_below_threshold = len(fn[fn < self.settings.frequency_threshold])

        return Sxx[n_below_threshold:, :], fn[n_below_threshold:]

    def process_sound_file(self, file:str, folder:str):
        ''' Calculates index values at test bands,
            file: path to the file
            test_bands: list of 3-tuples with band name, a 2-tuple with the min and max frequencies, 
                and a boolean for filtered with a butterworth filter (True) or extracted from the spectrogram (False) frequencies
            valid_file: a function which operates on the site and stamp
        '''
        file_info = self.valid_file(folder, file)
        if not file_info:
            return None

        site, stamp = file_info
        wave = sp.open_wav(file, trim_start=self.settings.trim_file_start, channel=1)
        ret = []
        for parameter in self.fft_windows:
            pre_Sxx, tn, pre_fn, ext = spectrogram(wave.signal, wave.fs, window=self.settings.window, nperseg=parameter, noverlap=0)
            if self.settings.frequency_threshold:
                pre_Sxx, pre_fn = self.remove_below_threshold(pre_Sxx, pre_fn)

            for band_name, freq, filtered in self.test_bands:
                if filtered:
                    fltr = butter(5, freq, 'bandpass', output='sos', fs=wave.fs)
                    fltrd_sig = sosfilt(fltr, wave.signal)
                    fltr_Sxx, fltr_tn, fltr_fn, fltr_ext = spectrogram(fltrd_sig, wave.fs, window=self.settings.window, nperseg=parameter, noverlap=0)
                    if self.settings.frequency_threshold:
                        fltr_Sxx, fltr_fn = self.remove_below_threshold(fltr_Sxx, fltr_fn)

                    fn = fltr_fn
                    Sxx = fltr_Sxx
                else:
                    above = len(pre_fn) - len(pre_fn[pre_fn > freq[1]])
                    below = len(pre_fn[pre_fn < freq[0]])
                    fn = pre_fn[below:above]
                    Sxx = pre_Sxx[below:above]

                for index in self.indices_of_interest:
                    func = self.get_index_func(index)
                    p = self.series_definition(index, band_name, filtered, stamp, site, parameter, func, Sxx, fn)
                    ret.append(p)

        return ret

    def build_model(self, r_link, marine, text_options, df, factors:list):
        ''' text_options: tuple of index, filtered, band_name, cross_effect 
        '''
        index, flt, band_name, cross_effect = text_options
        r_df = r_link.convert_to_rdf(df)
        for factor in factors:
            r_link.change_col_to_factor(r_df, factor)

        print(f"{band_name} {flt}using {index}")
        path = f"output/{cross_effect} x Window Conditional Effects for {index.upper()} over {flt}{band_name.lower()} frequencies"
        model, effects = r_link.r_src.find_effects(r_df, index, str(path), marine=marine, iter=self.settings.iterations, warmup=self.settings.warmup)
        print(model)
        warnings = r_link.r_src.get_warnings()
        if warnings != r_link.null_value:
            print(warnings)

        return model, effects[-1]

    def get_index_values(self, input_file=None, output_file=None):
        n_processes = self.settings.n_processes
        if input_file:
            return self.unpickle_data(input_file=None)

        serieses:list[pd.DataFrame] = []
        for folder in Path(self.settings.data_location).iterdir():
            files  = list(folder.glob("*.wav"))
            with Pool(n_processes) as pool, tqdm(total=len(files), leave=False) as pbar:
                ret = [pool.apply_async(self.process_sound_file, args=(i, folder.name), callback=lambda _:pbar.update(1)) for i in files]
                res = [r.get() for r in ret]
                for r in res:
                    if r is not None:
                        for s in r:
                            serieses.append(s)

        if output_file:
            self.pickle_data(serieses, output_file)

        return serieses
