import matplotlib.pyplot as plt
import numpy as np

y1 = np.array([
    0.9998058877100278,
    0.9996231232493897,
    0.9994202867173135,
    0.999206055723159,
    0.9990284935643631,
    0.9988702756842456,
    0.9988702756842456,
    0.9988702756842452,
    0.9988702756842456,
    0.9988702756842452,
    0.9988702756842452,
    0.9988702756842456,
    0.99872868832451,
    0.9984220328219945,
    0.9980260687473977,
    0.9976676504582339,
    0.997289463709415,
    0.9969344396659355,
    0.996448806601916,
    0.995884046335505,
    0.9953815552079974,
    0.9950470665746427,
    0.9951704153072468,
    0.995220879722308,])

y2 = np.array([
    0.9997608571377943,
    0.9995143517454056,
    0.9992398146894469,
    0.9990866342879549,
    0.9987985958196062,
    0.9985447333767142,
    0.9983955829809651,
    0.9981944748158399,
    0.9979606071136072,
    0.9979606071136073,
    0.9979606071136072,
    0.9979606071136072,
    0.9979606071136072,
    0.9979606071136072,
    0.9979606071136073,
    0.9979606071136072,
    0.9979606071136075,
    0.9979606071136072,
    0.9977805589584392,
    0.9974270860117274,
    0.9968988783363033,
    0.976242473168888,
    0.9762851670184327,
    0.9765209127533667,
    0.9764829647976336,
    0.976170672586648,
    0.9761159889919532,
    0.9760143626431607,
    0.9759959326039045,
    0.9758348174582947,
    0.9760748559380666,
    0.9758517602834345,
    0.975506489553445,
    0.9752569929575977,
    0.9755075660798457,
    0.9757376973820319,
])

x1 = np.arange(1, len(y1)+1)
x2 = np.arange(1, len(y2)+1)

fig = plt.figure(dpi=130)
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'
ax = plt.subplot(1, 1, 1)
ax.plot(x1, y1, label='2N=12', color='blue', linewidth=2)
ax.plot(x2, y2, label='2N=18', color='orange')
ax.scatter(x1, y1, color='blue', s=40, marker='o')
ax.scatter(x2, y2, color='orange', s=40, marker='x')

ax1 = plt.gca()
mylinewidth = 1.5
ax1.spines['bottom'].set_linewidth(mylinewidth)
ax1.spines['left'].set_linewidth(mylinewidth)
ax1.spines['top'].set_linewidth(mylinewidth)
ax1.spines['right'].set_linewidth(mylinewidth)
plt.xlabel('Gates Applied', fontsize=18)
plt.ylabel('Fidelity', fontsize=18)
plt.xticks(fontsize=13)
yticks = [0.96, 0.97, 0.98, 0.99, 1.00]
yticks_name = [str(i) for i in yticks]
plt.yticks(yticks, yticks_name, fontsize=13, fontname='Times New Roman')
plt.ylim(bottom=0.96)
plt.title('Fidelity over iterations', fontsize=18)
plt.legend()
plt.grid()
plt.show()
