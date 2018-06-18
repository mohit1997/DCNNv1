import numpy as np
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
import scipy.signal as signal
from peakdetect import peakdetect
import glob
import os
import pandas as pd


def getpeaks(path: str, one_hot=False):
    rate, wav = wf.read(path)
    data = np.array(wav)
    degg = np.insert(np.diff(data), 0, 0)  # To preserve no. of inputs
    out = np.array(peakdetect(degg, lookahead=5))
    print(np.array(out[1]).shape)
    out = np.array(out[1])[:, 0]

    # Soft threshold
    thresh = -1 / 6 * np.max(np.abs(degg))

    s = pd.Series(np.abs(degg))
    thresh = -1/6 * s.nlargest(100).mean()

    # Apply Threshold
    dec = degg[out] <= thresh
    fin = out[np.nonzero(1 * dec)]

    diff_fin = np.diff(fin)
    threshold = 50
    thresholded_diff = (diff_fin >= threshold) * 1.0
    final_diff = np.insert(thresholded_diff, len(
        thresholded_diff) - 1, 1) * fin

    fin = final_diff.astype(np.int)

    if one_hot:
        gt = np.zeros(len(degg))
        print("ONE HOT")
        gt[fin] = 1
        return fin, gt, degg

    return fin, degg[fin], degg


path = 'applawd_raw/c2'
outdir = 'applawd_raw/GT'

globpath = os.path.join(path, '*.wav')
files = sorted(glob.glob(globpath))

for f in files:
    if f != "applawd_raw/c2/as03a9.wav":
        continue
    _, gt, _ = getpeaks(f, one_hot=True)
    basename = os.path.basename(f)
    basename = os.path.splitext(basename)[0]
    outname = os.path.join(outdir, basename)
    np.save(outname, gt)
    # break
