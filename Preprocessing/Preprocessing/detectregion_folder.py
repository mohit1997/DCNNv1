import numpy as np
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
from glob import glob
import os
import re
import pandas as pd

def get_all_backup_paths(dir, level):
   pattern = dir + level * '/*'
   return [d for d in glob(pattern) if os.path.isdir(d)]


def mysort(l):
    def convert(t): return int(t) if t.isdigit() else t

    def key(k): return [convert(c) for c in re.split('([0-9]+)', k)]
    return sorted(l, key=key)


def getregions(path: str):
    rate, wav = wf.read(path)
    degg = np.insert(np.diff(wav), 0, 0)
    orig = np.array(wav)
    thresh = -1 / 6 * np.max(np.abs(degg))  # Use heuristic of 1/9 and 1/6 for applawd

    s = pd.Series(np.abs(degg))
    thresh = -1/6 * s.nlargest(100).mean()

    # Select points with value less than threshold, to detect points near gci showing the voiced regions
    out = degg < thresh
    samples = np.nonzero(out * 1.0)
    locn = np.array(samples).reshape(-1)
    # insert very high value, since it has to be the last voiced regional point
    dist = np.insert(np.diff(locn), len(locn) - 1, 10000)
    # Two voice regions have to be atleast 1000 samples apart
    dec = np.array(np.nonzero(1.0 * (dist > 1000))).reshape(-1)
    end = locn[dec]  # Ending positions of Voiced regions
    # one point after in the thresholded list locations will give starings except at the  last point
    start = locn[dec[:-1] + 1]
    # Insert the first thresholded point as starting of first voiced regions
    start = np.insert(start, 0, locn[0])

    return start, end, degg, np.array(wav)

def process_peaks(peak_locations, sig_size):
    diff_fin = np.diff(peak_locations)
    threshold = 50
    thresholded_diff = (diff_fin >= threshold) * 1.0
    final_diff = np.insert(thresholded_diff, len(
        thresholded_diff) - 1, 1) * peak_locations

    fin = final_diff.astype(np.int)
    fin = fin[np.nonzero(fin)]

    GT = np.zeros(sig_size)
    GT[fin] = 1
    return GT



def generate(path_speech: str, path_egg: str, np_egg: str, neg_shift: int, outdir_sp: str, outdir_pk: str, outdir_egg: str):
    _, wav = wf.read(path_speech)
    speech = np.array(wav)
    peaks: np.ndarray = np.load(np_egg)

    # print(speech.shape)
    # print(peaks.shape)

    shifted_speech = np.roll(speech, -abs(neg_shift))
    start, end, _, egg = getregions(path_egg)

    final_speech = [np.array(shifted_speech[st:en])
                    for st, en in zip(start, end)]
    final_peaks = [np.array(peaks[st:en]) for st, en in zip(start, end)]

    final_egg = [np.array(egg[st:en]) for st, en in zip(start, end)]

    my_speech = np.concatenate(final_speech, axis=0)
    my_peaks = np.concatenate(final_peaks, axis=0)
    my_egg = np.concatenate(final_egg, axis=0)

    ######
    my_pks = process_peaks(np.nonzero(my_peaks)[0], len(my_peaks))
    

    basename = os.path.basename(np_egg).split('.')[0]
    # print(my_speech.shape)
    # print(my_peaks.shape)

    p_speech = os.path.join(outdir_sp, basename)
    p_peaks = os.path.join(outdir_pk, basename)
    p_egg = os.path.join(outdir_egg, basename)

    np.save(p_speech, my_speech)
    np.save(p_peaks, my_pks)
    # np.save(p_egg, my_egg)


outdir_peaks = 'bdl_peaks'
outdir_egg = 'egg'

for f in [outdir_egg, outdir_peaks]:
    os.makedirs(f, exist_ok=True)


pathegg = 'applawd_raw/c2'
eggfiles = sorted(glob(os.path.join(pathegg, '*.wav')))
pathspeech = 'noise_speech_bdl/0/white'
speechfiles = mysort(glob(os.path.join(pathspeech, '*.wav')))
pathgt = 'applawd_raw/GT'
gtfiles = sorted(glob(os.path.join(pathgt, '*.npy')))


out_speech = sorted(get_all_backup_paths('noise_out_raw_applawd', level=2))
in_speech = sorted(get_all_backup_paths('noise_speech_raw_applawd', level=2))

print(out_speech, in_speech)

i = 0
for speechfiles, outdir_speech in zip(in_speech, out_speech):
    speechfiles = mysort(glob(os.path.join(speechfiles, '*.wav')))
    for f1, f2, f3 in zip(speechfiles, eggfiles, gtfiles):
        print(f1, f2, f3)
        generate(f1, f2, f3, 13, outdir_speech,
                outdir_peaks, outdir_egg)

        i = i + 1
