from dataclasses import dataclass

@dataclass
class CommonSettings:
    window:str = 'hamming'
    n_processes:int = 10
    frequency_threshold:int = 100
    trim_file_start:int = 5
    iterations:int = 10000
    warmup:int = 5000
    colors:tuple = None
    den:int = None
    num:int = None
    bands:dict = None
    sample_rate:int = None
    data_location:str = None
    round_to_minute:int = None
    cross_effect:str = None
