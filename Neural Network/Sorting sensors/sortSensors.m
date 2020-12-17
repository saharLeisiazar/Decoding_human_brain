
filename = sprintf('NeuroMagSensorsDeviceSpace.mat'); % Downloaded from Kaggle github
                                                      % https://github.com/FBK-NILab/DecMeg2014/tree/master/additional_files
data = load(filename);
pos_data= data.pos;
for i = 1 : 306
    pos(i, 1) = i
    pos(i, 2) = pos_data(i, 2)
end

sortedPos = sortrows(pos,2)

for i = 1 : 306
    SensorsSorted(i, 1) = sortedPos(i, 1)
end
save('D:/My Lessons/Machine Learning Project/Code/SensorsSortedLocation.mat','SensorsSorted');