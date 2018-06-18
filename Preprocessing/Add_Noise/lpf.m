function yh = lpf(y,cf,fs)


y=[y ; zeros(1,length(y))'];

 if mod(length(y),2)~=0
        
        y=[y ; 0];
        
 end
    
    
freq_res= fs/length(y);

    nn1= floor(cf/freq_res);
    
    
    %nn2= floor(50/freq_res);
    
    h1= 0.5+0.5*cos(pi.*(1:nn1)/(nn1));
    
%     h11=0.5+0.5*cos(pi.*(1:nn2)/(nn2));
%     
%     h1=[h1 h11];

    h2= zeros( 1, floor(length(y)/2)-length(h1));
    h3= [ h1 h2];
    h4= wrev(h3);
    h5= [ h3 h4];
    
 %  h5=h5.^4;

    g3= fft(y);
    
    
    sig_hp1= h5' .* g3;
    sig_hp1 = ifft (sig_hp1);
    yh= real(sig_hp1);
yh=yh(1:floor(length(y)/2));
%     [b,a] = butter(4,cf/fs,'low');
%     
%     yh=filter(b,a,y);