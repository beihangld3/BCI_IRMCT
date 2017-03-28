function EEG = eeg_car(EEG)

% Edited by M. Tavella <michele.tavella@epfl.com> on 06/04/09 22:23:38
%
% function EEG = eeg_car(EEG)
% Where EEG is a [samples x channels] matrix

% Receives a matrix time x channels and applies common average reference
% spatial filter for all time points

EEG = EEG - mean(EEG, 2) * ones(1, size(EEG, 2));
