function yn = add_noise(inp, type, snr)
y = inp;
if strcmp(type, 'white')
    noise= normrnd(0,1,[1 length(y)]);
    noise = noise';
end
if strcmp(type, 'babble')
    [wav ~] = audioread('babble.wav');
    noise = wav(1:length(y));
end

y= y ./ (norm(y));
sigp= sqrt(sum(y.*y)/length(y));
sigma=10^( -0.05*snr) *sigp;
noisep= sqrt(sum(noise.*noise)/length(noise));
noise= noise *(sigma / noisep); 
noisep1= sqrt(sum(noise.*noise)/length(noise));
snr_out= 20*log10(sigp / noisep1)
yn= y+ noise;
end
