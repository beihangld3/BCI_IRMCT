% [0.3081;0.2800;0.2925;0.2904;0.2857;0.2530;0.2918;0.2836;0.2775;0.3646];
figure(1)
imagesc(DPM12);
xlabel('Frequency band (Hz)');
ylabel('Channel label');
channel16 = importdata('channelLabels16.txt')'; 
set(gca,'YTick',1:16)
set(gca,'YTickLabel',channel16)
set(gca,'XTick',1:4:23)
set(gca,'XTickLabel',4:8:48)
set(gca, 'FontSize', 13)

figure(2)
topoplot(mean(DPM12(:,[4 5]),2),chanlocs16);

figure(3)
topoplot(mean(DPM12(:,[10 11]),2),chanlocs16);

