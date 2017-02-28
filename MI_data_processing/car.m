function signal = car(signal)

% Receives a matrix time x channels and applies common average reference
% spatial filter for all time points

signal = signal - mean(signal, 2) * ones(1, size(signal, 2));