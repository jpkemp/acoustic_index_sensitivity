from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from overrides import overrides
from tools.index_sensitivity import IndexSensitivity
from study_settings.common import CommonSettings

@dataclass
class BigVickySettings(CommonSettings):
    fish_frequencies:tuple = (200, 800)
    shrimp_frequencies:tuple = (2000, 5000)
    sample_rate:int = 48000
    data_location:str = "~/code/acoustic_analysis/data/big_vick"
    colors:tuple = ('#56B4E9', '#E69F00')
    num:str = "12"
    den:str = "0"
    round_to_minute:int = 15
    cross_effect:str = "Hour"

class BigVickyToolbox(IndexSensitivity):
    @overrides
    def check_file(cls, folder:Path, file:Path):
        '''retrieve site and timestamp information from the folder and filenames'''
        site, stamp, _ = file.name.split(".")
        stamp = IndexSensitivity.get_rounded_timestamp(stamp, BigVickySettings.round_to_minute, "%y%m%d%H%M%S")
        if stamp.hour != int(BigVickySettings.num) and stamp.hour != int(BigVickySettings.den):
            return False

        return site, stamp

    @overrides
    def series_definition(cls, index, band_name, filtered, stamp, site, parameter, func, Sxx, fn, truncation):
        '''series creation for rows of the dataframe, including column names, which vary slightly between the Carara and Big Vicky experiments'''
        ret = pd.Series([index, band_name, filtered, stamp.day, stamp.hour, site, parameter, func(Sxx, fn), truncation],
            index=["Index", "Band", "Filtered", "Day", "Hour", "Site", "Window", "Value", "Truncation"])

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
            cols_to_group = [x for x in df.columns if x != "Value"]
            df = df.groupby(cols_to_group).mean().reset_index()

        df["Site"] = df["Site"].astype(int)
        df["Hour"] = df["Hour"].astype(int)
        df["Window"] = df["Window"].astype(int)
        df["Value"] = df["Value"].astype(float)
        if output_path:
            df.to_pickle(output_path)

        return df    
