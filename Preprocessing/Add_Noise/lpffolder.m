addpath('/home/mohit/Interspeech/Prathosh_codes/fwd/')
indir = '/home/mohit/Interspeech/DataBaseOriginal/slt/cmu_us_slt_arctic/down/c1/*.wav';
outdir = '/home/mohit/Interspeech/DataBaseOriginal/slt/cmu_us_slt_arctic/down/c1/lowpass/';
files = (dir(indir));
[~, idx] = sort({files.name});
files = files(idx);
%out(1);
%out(1).name;
%out(1).folder;
%file = char(string(out(1).folder) + string('/') + string(out(1).name));

cf = 500;

for i =1:length(files)
    out = files(i);
    file = char(string(out.folder) + string('/') + string(files(i).name)) 
    [wav fs] = audioread(file);
    lwav = lpf(wav, cf, fs);
    whereto = char(string(outdir) + string(files(i).name));
    audiowrite(whereto, lwav, fs);
end