import unittest
import math
import numpy as np
from tools.sound_processor import SoundProcessor, WAV_MAX, WAV_MIN

class TestStringMethods(unittest.TestCase):
    def setUp(self):
        self.sp = SoundProcessor()

    def test_aci(self):
        # see https://ljvillanueva.github.io/soundecology/ACIandSeewave.html
        sound = self.sp.open_wav("data/tropicalsound.wav")
        spectro = self.sp.create_spectrogram(sound.signal, sound.fs, mode='magnitude')
        aci = self.sp.calculate_aci(sound, 5, spectro=spectro)
        total_aci = aci
        self.assertEqual(int(total_aci), 660)

    def test_sine(self):
        sound = self.sp.create_sine(1000, 3, 2010) # unsure why this fails at Nyquist freq. Perhaps a sample time rounding issue?
        f, t, sxx = self.sp.create_spectrogram(sound.signal, sound.fs)
        idx = np.argmax(sxx, axis=0)
        self.assertTrue(all(x == idx[0] for x in idx))
        self.assertGreater(f[idx[0] + 1], 1000)
        self.assertLess(f[idx[0] - 1], 1000)

    def test_open_wav(self):
        file_path = "test/test_sine.wav"
        fs = 2010
        for data_format in [np.int16, np.int32]:
            amp = np.iinfo(data_format).max
            test_sound = self.sp.create_sine(1000, 3, fs, amp=amp, output_path=file_path) 
            sound = self.sp.open_wav(file_path, normalise=False)
            self.assertEqual(len(sound.signal), len(test_sound.signal))
            for i, v in enumerate(test_sound.signal):
                self.assertEqual(v, sound.signal[i])

            self.assertEqual(fs, sound.fs)

if __name__ == '__main__':
    unittest.main()