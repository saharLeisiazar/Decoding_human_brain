import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# set width of bar
barWidth = 0.1
padding=0.02
plt.figure(figsize=(15,10))
# set height of bar


barsSVM = [83.57, 88.48, 89.64,90.06,90.35]
barsCNN = [81.57 , 85.4 , 90.23  ,  94.11 , 92.19]
barsFC = [82.37 , 88.85 , 91.72  , 95.5 , 95.33]
barsET = [83.37, 88.2, 90.2, 91.44,90.49]
barsRF = [81.57, 87.48, 89.15, 90.7,88.68]


# Set position of bar on X axis
r1 = np.arange(len(barsSVM))
r2 = [x +padding+ barWidth for x in r1]
r3 = [x +padding+ barWidth for x in r2]
r4 = [x +padding+ barWidth for x in r3]
r5 = [x +padding+ barWidth for x in r4]
# Make the plot
ax = plt.gca()
ax.grid(which='major', axis='y', linestyle='-')
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=18)
plt.bar(r1, barsSVM, color='#d1eb10', width=barWidth, edgecolor='white', label='SVM')
plt.bar(r2, barsCNN, color='#8dd3c7', width=barWidth, edgecolor='white', label='CNN')
plt.bar(r3, barsFC, color='#80b1d3', width=barWidth, edgecolor='white', label='FC')
plt.bar(r4, barsET, color='#bebada', width=barWidth, edgecolor='white', label='Extra Tree')
plt.bar(r5, barsRF, color='#fb8072', width=barWidth, edgecolor='white', label='Random Forest')
# Add xticks on the middle of the group bars
# matplotlib.rcParams.update({'font.size': 16})
plt.xlabel('Number of features', fontweight='bold',fontsize=16)
plt.ylabel('Accuracy (%)', fontweight='bold',fontsize=16)
plt.xticks([r + 0.12+padding+barWidth for r in range(len(barsSVM))], ['10', '20', '30', '40','50'])
plt.yticks([10,20,30,40,50,60,70,75,80,85,90,95,100],['10','20','30','40','50','60','70','75','80','85','90','95','100'])
matplotlib.rc('xtick', labelsize=22)
matplotlib.rc('ytick', labelsize=40)
# Create legend & Show graphic

plt.legend(prop={"size":11,'weight':'bold'})
plt.show()
