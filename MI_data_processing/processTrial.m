function psd = processTrial(trial, sp_filter)

% Receives an MI trial time x channels and computes PSD features after 
% preprocessing on a running window for all channels. 
% It returns a matrix time x channels x frequency. 
% This code is optimized for understanding and certainly not for efficiency

fs = 512;
winsec = 1;
winsize = winsec*fs;
winshiftsec = 0.0625;
winshift = winshiftsec*fs;

psdwinsec = 0.5;
psdwin = psdwinsec*fs;
psdovl = 0.5;

freqs = [4:2:48];
pos = 1;
sampleidx=0;

while(pos+winsize-1 <= size(trial,1))
    
    sampleidx = sampleidx + 1;
    
    % Get current part of raw signal for this trial
    sample = trial(pos:pos+winsize-1,:);
    
    % Remove DC
    sample = removeDC(sample);
    
    % Apply spatial filter
    %sample = car(sample);
    if strcmp(sp_filter,'CAR')
        sample = car(sample);
    elseif strcmp(sp_filter,'Laplacian')
        sample = laplacianSP(sample);
    elseif strcmp(sp_filter,'Laplacian')
        sample = sample;
    end
    
    % Extract PSD 
    disp(['Calculating sample: ' num2str(sampleidx)]);
    
    for ch=1:size(trial,2)
        psd(sampleidx,ch,:) = extractPSD(sample(:,ch),psdwin,psdovl, freqs, fs);
    end
    
    pos = pos + winshift;
    
end
