import re
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
from overrides import overrides
from tools.index_sensitivity import IndexSensitivity
from study_settings.common import CommonSettings

@dataclass
class CararaSettings(CommonSettings):
    sample_rate:int = 22050
    max_freq:int = sample_rate / 2
    data_location:str = "~/code/acoustic_analysis/data/terra/2014"
    colors:tuple = ("#F0E442", '#009E73')
    num:str = "HW"
    den:str = "IR"
    round_to_minute:int = 5
    cross_effect:str = "Site"

class CararaToolbox(IndexSensitivity):
    @overrides
    def check_file(cls, folder:Path, file:Path):
        '''retrieve site and timestamp information from the folder and filenames'''
        site = re.findall('([a-zA-Z ]*)\d*.*', folder.name)[0]
        stamp = re.findall("(\d+_\d+).wav", file.name)[0]
        stamp = IndexSensitivity.get_rounded_timestamp(stamp, CararaSettings.round_to_minute, "%Y%m%d_%H%M%S")

        return site, stamp

    @overrides
    def series_definition(cls, index, band_name, filtered, stamp, site, parameter, func, Sxx, fn):
        '''series creation for rows of the dataframe, including column names, which vary slightly between the Carara and Big Vicky experiments'''
        ret = pd.Series([index, band_name, filtered, stamp, site, parameter, func(Sxx, fn)],
            index=["Index", "Band", "Filtered", "Time", "Site", "Window", "Value"])

        return ret

    @overrides
    def create_dataframe(cls, serieses:list, input_path:str=None, output_path:str=None):
        '''concatenates a list of series to a dataframe, and converts columns to the correct types
        
        input_path: if given, loads an existing dataframe before setting the types, instead of concatenating a series
        output_path: if given, saves the dataframe to file'''
        if input_path:
            df = IndexSensitivity.unpickle_data(input_path)
        else:
            df = pd.concat(serieses, axis=1).transpose()

        df["Time"] = pd.to_datetime(df["Time"])
        df["Window"] = df["Window"].astype(int)
        df["Value"] = df["Value"].astype(float)

        if output_path:
            df.to_pickle(output_path)

        return df