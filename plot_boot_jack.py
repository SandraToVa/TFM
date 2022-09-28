#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preview'] = True
from matplotlib.backends.backend_pdf import PdfPages

with open('EMP_prot_boot.dat', 'r') as f:
    data = f.read()

data = data.split('\n')[:-1]
xboot = [float(row.split()[0]) for row in data]
yboot = [float(row.split()[1]) for row in data]
eboot = [float(row.split()[2]) for row in data]

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size='12')

fig = plt.figure(figsize=(8,6))

plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.95, wspace=0.21, hspace=0.2)

#########################
fig1 = fig.add_subplot(1,1,1)

#fig1.set_title("Effective mass plot")
fig1.set_ylabel(r'$\mathrm{m} \,\mathrm{(l.u.)}$')
fig1.set_xlabel(r'$t \,\mathrm{(l.u.)}$')
fig1.set_ylim([1,1.3])
fig1.set_xlim([0,20.5])
#plt.xticks([5,10,15,20])
plt.minorticks_on()
fig1.axes.tick_params(which='both',direction='in')
fig1.yaxis.set_ticks_position('both')
fig1.xaxis.set_ticks_position('both')

fig1.errorbar(xboot,yboot, yerr=eboot, c='#ED553B', ls='None', marker='o', markersize=6, capsize=1, elinewidth=0.7,label="Bootstrap")
#fig1.errorbar(xjack,yjack, yerr=ejack, c='#20639B', ls='None', marker='o', markersize=6, capsize=1, elinewidth=0.7,label="Jackknive")


plt.legend()

#plt.show()

with PdfPages('EMP_prot_boot.pdf') as pdf:
   pdf.savefig(fig)
