function signal = removeDC(signal)

% Receives a matrix time x channels and removes the DC for all channels

signal = signal - repmat(mean(signal,1),size(signal,1),1);