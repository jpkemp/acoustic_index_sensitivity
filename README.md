# Acoustic Index Parameter Sensitivity Analysis
This repository stores the code to reproduce a sensitivity analysis conducted on several common ecoacoustic indices.

The main code and results are presented in Jupyter notebooks.

The file 'aci.ipynb' explores the effect of parameter selection on the Acoustic Complexity Index, by adjusting the j_bin and FFT window parameters. ([[1]](#1))

The file 'within_study_test.ipynb' explores the effect of parameter selection on four common acoustic indices: Acoustic Complexity Index, Acoustic Diversity Index, Acoustic Evenness Index, and Bioacoustic Index on two datasets, one from Santa Rosa National Park in Costa Rica and one from Big Vicky's Reef in Australia ([[1]](#1),[[2]](#2),[[3]](#3) ).

The file 'posterior_checks'.ipynb presents the denisty overlay and scatter average posteriro checks for the within-study test.

The file 'frequency_impact.ipynb' explores the effect of frequency on the interaction between simulated calls and FFT window parameter selection.


## References
<a id="1">[1]</a>
N. Pieretti, A. Farina, and D. Morri. A new methodology to infer the singing activity of an avian community: the Acoustic Complexity Index (ACI). Ecological Indicators, 11(3):868–873, 2011. doi: 10.1016/j.ecolind.2010.11.005.

<a id="2">[2]</a>
Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011. A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.DOI: 10.1007/s10980-011-9636-9

<a id="3">[3]</a>
Boelman NT, Asner GP, Hart PJ, Martin RE. 2007. Multi-trophic invasion resistance in Hawaii: bioacoustics, field surveys, and airborne remote sensing. Ecological Applications 17: 2137-2144. DOI: 10.1890/07-0004.1
