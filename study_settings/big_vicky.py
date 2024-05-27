from dataclasses import dataclass
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
