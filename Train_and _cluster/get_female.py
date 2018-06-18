import os
import glob
from shutil import copyfile

outdir = 'aplawd_female_peaks'
if not os.path.exists(outdir):
    os.makedirs(outdir)

indir = "aplawd_peaks"

dir = "aplawd_peaks/*.npy"
lis = glob.glob(dir)

speakers = ['f', 'g', 'h', 'i', 'j']
# print(len(lis))

# mod_lis = [i.replace('egg_', '') for i in lis]
mod_lis = lis

# for k in mod_lis:
# 	print(os.path.basename(k))

# for i in range(len(mod_lis)):
# 	os.rename(lis[i], mod_lis[i])

new_lis = []
for j in speakers:
	new_lis += [i for i in mod_lis if os.path.basename(i).find(j) is not -1]

dest = [os.path.join(outdir, os.path.basename(l)) for l in new_lis]
src = [os.path.join(indir, os.path.basename(l)) for l in new_lis]

for i in range(len(dest)):
	copyfile(src[i], dest[i])



# print((new_lis))

