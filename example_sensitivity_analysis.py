from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tools.sound_processor import SoundProcessor as sp

# select input recordings
recordings_category_a = ["data/tropicalsound.wav"]
recordings_category_b = ["data/tropicalsound.wav"]

# select acoustic index
index = sp.calculate_aci

# select appropriate parameters
nfft = [256, 512, 1024]
window_shape = "hamming"
j_bin = [5, 10, 20]

all_params = [nfft, j_bin]
parameter_combinations = list(product(*all_params))

# run sensitivity analysis
results = {}
for filename in recordings_category_a + recordings_category_b:
    results[filename] = {}
    sound = sp.open_wav(filename)
    for params in parameter_combinations:
        spectro = sp.create_spectrogram(sound.signal, 
                                        sound.fs, 
                                        samples=params[0], 
                                        window=window_shape) 
        aci = index(sound, params[1], spectro)
        results[filename][params] = aci

# plot results
outputs = []
for files in [recordings_category_a, recordings_category_b]:
    output = [pd.Series(v).reset_index() for k, v in  results.items() if k in files]
    output = pd.concat(output).groupby(['level_0', 'level_1']).mean()
    outputs.append(output)


lgd_names = ["Category A", "Category B"]
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
for i, output in enumerate(outputs):
    ustack = output.unstack()
    x, y = np.meshgrid(nfft, j_bin)
    z = output[0].unstack().to_numpy()

    ax.plot_surface(x, y, z, label=lgd_names[i])

ax.legend()
ax.set_xlabel("NFFT (samples)")
ax.set_ylabel("j_bin (s)")
ax.set_zlabel("ACI")
fig.savefig("test.png")
    
# should possibly do table instead; more robust to additional dimensions