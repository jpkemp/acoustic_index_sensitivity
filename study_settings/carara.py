from dataclasses import dataclass
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
