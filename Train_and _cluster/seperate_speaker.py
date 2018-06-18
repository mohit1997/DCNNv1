import numpy as np 
import glob 
import os

from shutil import copyfile

in_speech = 'aplawd_speech_raw'
in_peaks = 'aplawd_peaks'

speech_files = glob.glob(os.path.join(in_speech, '*.npy'))
peak_files = glob.glob(os.path.join(in_peaks, '*.npy'))

out_speaker = 'aplawd_speakers'

# test_speech = 'aplawd_test/speech'
# test_peaks = 'aplawd_test/peaks'

# outdir = [train_speech, train_peaks, test_speech, test_peaks]

# for j in outdir:
# 	os.makedirs(j, exist_ok=True)
	# os.system('rm -rf ' + str(j) + '/')

males = ['a', 'b', 'c', 'd', 'e']
females = ['f', 'g', 'h', 'i', 'j']

train = 3

train_names = males[:train] + females[:train]
test_names = males[train:] + females[train:]

for i in speech_files:
	file = os.path.basename(i)
	print(file)
	locn = file[-6]

	src_speech = os.path.join(in_speech, file)
	src_peaks = os.path.join(in_peaks, file)

	dest_speech = os.path.join(out_speaker, locn, 'speech', file)
	dest_peaks = os.path.join(out_speaker, locn, 'peaks', file)

	os.makedirs(os.path.join(out_speaker, locn, 'speech'), exist_ok=True)
	os.makedirs(os.path.join(out_speaker, locn, 'peaks'), exist_ok=True)
	# print(dest_peaks)
	copyfile(src_speech, dest_speech)
	copyfile(src_peaks, dest_peaks)

	# else:
	# 	dest_speech = os.path.join(test_speech, file)
	# 	dest_peaks = os.path.join(test_peaks, file)
	# 	copyfile(src_speech, dest_speech)
	# 	copyfile(src_peaks, dest_peaks)
