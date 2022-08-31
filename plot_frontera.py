#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import math
import itertools
#import random #MI: millor fes servir np.random.normal
import matplotlib.patches as patches
from matplotlib.patches import Polygon

#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preview'] = True
from matplotlib.backends.backend_pdf import PdfPages

import h5py

nsc=454  #N # Utilitza nsc=454 si fas servir les dades del fitxer h5
nt=19
nboot=nsc    #Nb
nsc_=1./nsc
nsc1_=1./(nsc-1)
nboot_=1./nboot
nbot_=1./(nboot-1)
#Creo un bucle per a plotejer totes les dades juntes
#Numero de operadors n_op
#n_op=10  #10
n_oplist = [3,4,5,6]
n_op = len(n_oplist)
n_smear = 1
ndim = n_op*n_smear
ndim_t=4
yboot=[]
eboot=[]
#op=3,11
#DELS FITXER .H5
fh5 = h5py.File('/Users/sandra/Documents/GitHub/TFM/qblocks_matrix_irreps_cl3_32_48_b6p1_m0p2450_frontera.h5', 'r')
#fh5 = h5py.File('/Volumes/GoogleDrive-100962958533814099960/.shortcut-targets-by-id/1mV9ITzWE4JpeOWAX6GdzvZ5l3if1bz8j/Sandra Tomás/Codis/data/qblocks_matrix_irreps_cl3_32_48_b6p1_m0p2450_frontera.h5', 'r')
blck = np.zeros((nsc,ndim,ndim,nt))
for i,opsrc in enumerate(n_oplist):
    for j,opsnk in enumerate(n_oplist):
        for ii in range(n_smear):
            for jj in range(n_smear):
                iii = i+ii*n_op
                jjj = j+jj*n_op
                blck[:,iii,jjj,:] = 0.5*(np.real(np.array(fh5['B2_I1_A1_f'][0:nsc,opsrc,ii,opsnk,jj,0:nt])+np.real(np.array(fh5['B2_I1_A1_b'][0:nsc,opsrc,ii,opsnk,jj,0:nt]))))

blck2=np.zeros((nsc,ndim,ndim,nt))
for op_src in range(ndim):
    for op_snk in range(ndim):
        #Var1 pre-processing antes de diagonalitzar C_ab(t)
        for k in range(nt):
            blck2[:,op_src,op_snk,k]=0.5*(blck[:,op_src,op_snk,k]+blck[:,op_snk,op_src,k])/np.sqrt(blck[:,op_src,op_src,0]*blck[:,op_snk,op_snk,0])

#Var1.2 Les matrius C_b(t)=pmeanboot correlator samples bootstrap samples NxN=[opsrc][opsnk]
#Millor passar array de op especifics
pmeanboot=np.zeros((nboot,ndim,ndim,nt))
x=np.random.uniform(size=(nsc,nboot))  #Matriu de num aleatoris entre 0 i 1
for j in range(nboot):
    boot=np.zeros((ndim,ndim,nt))
    for i in range(nsc):
        boot=boot+blck2[int(x[i][j]*nsc),:,:,:]
    pmeanboot[j,:,:,:]=boot*nsc_   #Ara hem generat les Nb bootstrap samples Cb(t)

#pmeanboot=C(t)
#C=\tilde{C}(t)
#C_diag=matriu final C_alpha,alpha que utilitzem per trobar la energia
#Var3.a find the cholesky decomposition of pmeanboot2(t0)
t_0=4-1 #t0=4
#Tot per t_ref=2*t_0=8
t_ref=2*(t_0+1)-1   #El -1 es per la llista q comensa en 0
C_diag=np.zeros((nboot,ndim,ndim,nt))
for n in range(nboot):
    L=np.linalg.cholesky(pmeanboot[n,:,:,t_0])
    L_=np.linalg.inv(L) #Matriu inversa
    Lt=L.T   #Matriu adjunta
    L_t=np.linalg.inv(Lt)   #invers de l'adjunt

    #Var3.b Calulem \tilde{C}(t)=C(t)
    matrix=np.matmul(L_,pmeanboot[n,:,:,t_ref])
    C=np.matmul(matrix,L_t)

    #Var3.c diagonalitzar C i trovar vectors propis \tilde{u}=v i valors propis w (q no utilitzo) per a t_ref
    w, v = np.linalg.eig(C)
    #Var3.d original eigenvectors u per a t_ref
    u=np.matmul(L_t,v)
    #La adjunta de los eigenvectors
    ut=u.T

    #Var3.e trobar la C_alpha,alpha(t)
    for k in range(0,nt):
        matrix=np.matmul(ut,pmeanboot[n,:,:,k])
        Cdiag=np.matmul(matrix,u)
        C_diag[n,:,:,k] = Cdiag

    #print(C_diag[n,:,:,k])

#3. Calculem Eb(t) per a los op iguals a la sink i a la source
for op in range(ndim):
    kt=1
    EMpoint=0.
    EMpoint=np.zeros((nboot,(nt-kt)))
    for k in range(0,(nt-kt)):
        for j in range(0,nboot):
            EMpoint[j][k]=np.log(C_diag[j,op,op,k]/C_diag[j,op,op,k+kt])/kt

    #Clculem \Bar{E}(t) i errors
    mean=np.zeros(nt-kt)
    for k in range(0,(nt-kt)):
        suma=0
        for j in range(0,nboot):
            suma=suma+EMpoint[j][k]
        mean[k]=suma*nboot_
    sigm=np.zeros(nt-kt)
    for k in range(0,(nt-kt)):
        sigma=0
        for j in range(0,nboot):
            sigma=sigma+(EMpoint[j][k]-mean[k])*(EMpoint[j][k]-mean[k])
        sigm[k]=np.sqrt(sigma*nboot_*nsc*nsc1_)
    #Dades
    yboot.append(mean)
    eboot.append(sigm)

xboot=list(range(1, nt))

m_n=1.203
levels_t=[2*m_n,2*math.sqrt(m_n**2+(2*math.pi/32)**2),2*math.sqrt(m_n**2+2*(2*math.pi/32)**2),2*math.sqrt(m_n**2+3*(2*math.pi/32)**2)] #2m_n, 2sqrt(m^2+(2*pi/32)^2])


plt.rc('text', usetex=True)
plt.rc('font', family='serif', size='12')

fig = plt.figure(figsize=(8,6))

plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.95, wspace=0.21, hspace=0.2)
##################            #No cal fer un gràfic cada vegada pero m'ajuda a visualitzar l'ajust -> Cal canviar-ho quan tot vagi be
#PLOT LINEAL
fig1 = fig.add_subplot(1,1,1)
fig1.set_title("Effective mass plot")
fig1.set_ylabel(r'$\mathrm{E} \,\mathrm{(l.u.)}$')
fig1.set_xlabel(r'$t \,\mathrm{(l.u.)}$')
fig1.set_ylim([2.380,2.550]) #1.10,1.3
fig1.set_xlim([0,19]) #0,20.5
#plt.xticks([5,10,15,20])
plt.minorticks_on()
fig1.axes.tick_params(which='both',direction='in')
fig1.yaxis.set_ticks_position('both')
fig1.xaxis.set_ticks_position('both')
#Plots dels diferents operadors
for i in range(ndim):
    fig1.errorbar(xboot,yboot[i], yerr=eboot[i], ls='None', marker='o', markersize=6, capsize=1, elinewidth=0.7)
for i in range(ndim_t):
    plt.plot([0,19],[levels_t[i],levels_t[i]],c='black',ls='--')
#plt.legend()
#plt.show()
with PdfPages('B2_I1_A1_plot.pdf') as pdf:
    pdf.savefig(fig)

plt.close('all')

#Plot of the energy levels despues de fer el fit amb plot_frontera_fit_plot.py
#nivells energia lineal i exponentcial
e_levels=[[2.406,2.406],[2.431,2.431],[2.462,2.462],[2.496,2.496]]
ebar=[[0.0048,0.0049],[0.0061,0.0061],[0.0062,0.0035],[0.0074,0.0075]]

e_levels_no3=[[2.431,2.431],[2.462,2.462],[2.496,2.496]]
ebar_no3=[[0.0058,0.0059],[0.0056,0.0057],[0.0066,0.0066]]

e_levels_no4=[[2.406,2.407],[2.461,2.461],[2.496,2.496]]
ebar_no4=[[0.0045,0.0074],[0.0065,0.0067],[0.0074,0.0075]]

e_levels_no5=[[2.4063349914550782,2.406726030809348],[2.432021427154541,2.432021770304856],[2.495771484375001,2.4957715528791518]]
ebar_no5=[[0.004399067671495806,0.004805618782495939],[0.0057401894057316685,0.005739931545644638],[0.006066538661732237,0.006080462341260557]]

x_levels=[1,2]
color=['tab:blue','tab:orange','tab:green','tab:red']


plt.rc('text', usetex=True)
plt.rc('font', family='serif', size='12')

fig = plt.figure(figsize=(3,6))

plt.subplots_adjust(left=0.30, bottom=0.08, right=0.98, top=0.95, wspace=0.21, hspace=0.2)
##################
fig1 = fig.add_subplot(1,1,1)
fig1.set_title("Energy levels")
fig1.set_ylabel(r'$\mathrm{E} \,\mathrm{(l.u.)}$')
#fig1.set_xlabel(r'$t \,\mathrm{(l.u.)}$')
fig1.set_ylim([2.380,2.550]) #1.10,1.3
fig1.set_xlim([0,3]) #0,20.5
fig1.set_xticklabels(['','Lin', 'Exp',''])
#plt.xticks([5,10,15,20])
plt.minorticks_on()
fig1.axes.tick_params(which='both',direction='in')
fig1.yaxis.set_ticks_position('both')
fig1.xaxis.set_ticks_position('both')
#Plots dels diferents operadors
for i in range(ndim):
    fig1.errorbar(x_levels[0],e_levels_no5[i][0], yerr=ebar_no5[i][0], ls='None', marker='v', c=color[i], markersize=6, capsize=1, elinewidth=0.7)
    fig1.errorbar(x_levels[1],e_levels_no5[i][1], yerr=ebar_no5[i][1], ls='None', marker='s', c=color[i], markersize=6, capsize=1, elinewidth=0.7)
for i in range(ndim_t):
    plt.plot([0,3],[levels_t[i],levels_t[i]],c='black',ls='--')
#plt.legend()
#plt.show()
with PdfPages('levels_no5.pdf') as pdf:
    pdf.savefig(fig)

plt.close('all')
