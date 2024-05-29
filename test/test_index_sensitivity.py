import os
import unittest
from datetime import datetime as dt
import numpy as np
from pathlib import Path
from study_settings.big_vicky import BigVickySettings, BigVickyToolbox
from study_settings.carara import CararaSettings, CararaToolbox
from study_settings.common import CommonSettings
from tools.index_sensitivity import IndexSensitivity
from tools.sound_processor import SoundProcessor 

class TestToolboxFunctions(unittest.TestCase):
    def setUp(self):
        window_sizes = [256, 512, 1024]
        test_bands = [("test_a", (200, 7000), True), 
                      ("test_b", (500, 9000), False), 
                      ("test_c", (100, 200), True), 
                      ("test_d", (200, 700), False)]
        indices_of_interest = ["ACI", "BIO"]
        
        self.vicky = BigVickyToolbox(BigVickySettings,
                                        window_sizes,
                                        test_bands,
                                        indices_of_interest)
        
        self.carara = CararaToolbox(CararaSettings,
                                    window_sizes,
                                    test_bands,
                                    indices_of_interest)
        self.sp = SoundProcessor()

    def test_rounding(self):
        fmt = "%Y%m%d%H%M%S"
        timestamps = ["20250515100729", "20250516100731"]
        nearest = [15, 5]
        expected = [0, 15, 5, 10]
        for i, near in enumerate(nearest):
            for j, stamp in enumerate(timestamps):
                new_stamp = IndexSensitivity.get_rounded_timestamp(stamp, near, fmt)
                self.assertEqual(new_stamp.minute, expected[i * 2 + j])

    def test_thresholding(self):
        fs = 2010
        snd = self.sp.create_sine(3000, 3, fs)
        f, t, sxx = self.sp.create_spectrogram(snd.signal, fs)
        for toolbox in [self.carara, self.vicky]:
            sxxn, fn = toolbox.remove_below_threshold(sxx, f)
            self.assertTrue(min(fn) >= CommonSettings.frequency_threshold)
            self.assertEqual(max(fn), max(f))
            self.assertEqual(len(fn), sxxn.shape[0])
            test_idx = np.where(f == fn[0])[0][0]
            self.assertEqual(sxx[test_idx, 0], sxxn[0, 0])

    def test_carara_folder(self):
        folder = Path("test/HW1-20240505T032601Z-001") 
        filename = folder / "BUSY_20140428_050300.wav"
        site, stamp = self.carara.check_file(folder, filename)
        self.assertEqual("HW", site)
        self.assertEqual(stamp, dt(2014,4,28,5,5))

    def test_vicky_folder(self):
        folder = Path("test/1207980063")
        filename = folder / "1207980063.221110142958.wav"
        valid = self.vicky.check_file(folder, filename)
        self.assertFalse(valid)
        filename = folder / "1207980063.221110124458.wav"
        site, stamp = self.vicky.check_file(folder, filename)
        self.assertEqual("1207980063", site)
        self.assertEqual(stamp, dt(2022,11,10,12,45))

    def test_extract_frequencies(self):
        folder = Path(os.path.expanduser(self.vicky.settings.data_location))
        filename = folder / "1207980063" / "1207980063.221118121458.wav"
        sound = self.sp.open_wav(str(filename),trim_start=5)
        f, t, sxx = self.sp.create_spectrogram(sound.signal, sound.fs)
        for band in [(500, 5000), (9000, 10000), (300, 400), (1, 2), (1400, 1450)]:
            sxxn, fnn = self.vicky.extract_frequencies_from_sxx(sxx, f, band)
            test_f = [x for x in f if x >= band[0] and x <= band[1]]
            test_not_f = [x for x in f if x < band[0] or x > band[1]]
            for freq in test_f:
                self.assertTrue(freq in fnn)

            for freq in test_not_f:
                self.assertTrue(freq not in fnn)

            self.assertEqual(len(fnn), sxxn.shape[0])
            if fnn.size == 0: continue
            test_idx = np.where(f == fnn[0])[0][0]
            self.assertEqual(sxx[test_idx, 0], sxxn[0, 0])
            test_idx = np.where(f == fnn[-1])[0][0]
            self.assertEqual(sxx[test_idx, 0], sxxn[-1, 0])

    def test_process_sound_file(self):
        folder = Path(os.path.expanduser(self.vicky.settings.data_location))
        filename = folder / "1207980063" / "1207980063.221118121458.wav"
        data = self.vicky.process_sound_file(filename, folder)
        self.assertEqual(len(data), 3 * 4 * 2) # 3 FFT windows, 4 bands, 2 indices
