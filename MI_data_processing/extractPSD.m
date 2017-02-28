function psd = extractPSD(signal, psdwin, psdovl, freqs, fs)

% Computes PSD of signal with given settings:
% psdwin: internal psdwindow in samples
% psdovl: overlapping as percentage in [0,1]
% freqs: desired frequency points
% fs: sampling rate

% Easy but slow
% [psd, frq] = pwelch(signal, psdwin, psdwin*psdovl , freqs, fs);

% Harder but faster
[psd, frq] = pwelch(signal, psdwin, psdwin*psdovl , [], fs);
[val, set] = intersect(frq, freqs);

if(length(set) ~= length(freqs))
	disp('[extractPSD] Warning: cannot provide requested frequencies!');
end

psd = psd(set);