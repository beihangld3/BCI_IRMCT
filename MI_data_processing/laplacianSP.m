function signal = laplacianSP(signal, lap)

% Receives a matrix time x channels and NSWE cross laplacian spatial 
% filter for all time points

if(nargin < 2)
    lap = load('laplacian16.mat');
    lap = lap.lap;
end

if(size(signal,2)~=size(lap,1))
    disp('[laplacianSP] Incompatible signal and laplacian matrix!')
    signal = [];
    return;
else
    signal = signal*lap;
end