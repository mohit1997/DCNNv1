addpath('/home/mohit/Interspeech/Prathosh_codes/fwd/')
indir = '/home/mohit/Interspeech/DataBaseOriginal/Applawd/APLAWD/c1/*.wav';
home = '/home/mohit/Interspeech/DataBaseOriginal/Applawd/APLAWD/noise_speech_raw/';
files = (dir(indir));
[~, idx] = sort({files.name});
files = files(idx);
%out(1);
%out(1).name;
%out(1).folder;
snr = 0;

%file = char(string(out(1).folder) + string('/') + string(out(1).name));

cf = 500;
for snr = 0:5:25 
    type = 'babble';
    outdir = char(string(home) + string('/') + string(snr) + string('/') + string(type) + string('/'));
    if ~exist(outdir)
        mkdir(outdir)
    end
    for i =1:length(files)
        out = files(i);
        file = char(string(out.folder) + string('/') + string(files(i).name)) 
        [wav fs] = audioread(file);
        nwav = add_noise(wav, type, snr);
        %lwav = lpf(nwav, cf, fs);
        whereto = char(string(outdir) + string(files(i).name));
        audiowrite(whereto, nwav, fs);
    end
end


for snr = 0:5:25 
    type = 'white';
     outdir = char(string(home) + string('/') + string(snr) + string('/') + string(type) + string('/'));
    if ~exist(outdir)
        mkdir(outdir)
    end
    for i =1:length(files)
        out = files(i);
        file = char(string(out.folder) + string('/') + string(files(i).name)) 
        [wav fs] = audioread(file);
        nwav = add_noise(wav, type, snr);
        %lwav = lpf(nwav, cf, fs);
        whereto = char(string(outdir) + string(files(i).name));
        audiowrite(whereto, nwav, fs);
    end
end