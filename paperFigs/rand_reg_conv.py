import numpy as np
import os
os.chdir('..')
from SISN_mp import *
os.chdir('paperFigs')
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
cols = ['#666666','#1b9e77','#e7298a','#7570b3',
        '#d95f02','#66a61e','#e6ab02','#a6761d']


f = lambda  b, k, N: k_regular(k, b, (k-1) * b * np.sqrt(N-1), N, max_its=2000)[0][0]

plt.figure(1,figsize=(2.0*1.618, 1.0*2.5))
cmap = plt.get_cmap('coolwarm')

B = np.linspace(0.45, 0.8, 500)

for K in range(1, 9):
    plt.plot(B, [f(b, 3, K) for b in B], c=cmap((K-1)/8), ls='-', label=rf"$K={K}$")
plt.xlim(0.45, 0.8)
plt.xlabel(r"infection rate, $\beta$")
plt.ylabel(r"fraction infected, $I$")
plt.ylim(0.0, 0.5)
plt.legend(ncol=2, fontsize='small', handlelength=0.8, handletextpad=0.5, columnspacing=1.0)
plt.tight_layout()
plt.savefig('conv.pdf')


