#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sympy import Symbol, Matrix, sqrt

C1 = Symbol('c1')
C2 = Symbol('c2')
C3 = Symbol('c3')
C4 = Symbol('c4')
C5 = Symbol('c5')
C6 = Symbol('c6')

#All N-N states

pp = Symbol('pp')
pn = Symbol('pn')
np = Symbol('np')
nn = Symbol('nn')
NNstates = Matrix( [pp, pn, np, nn] )

NNJ0 = Matrix( [ [2 * (C1 - C2  + C5 - C6 ),0,0,0] , [0,C1 - C2 + C5 - C6, C1 - C2 + C5 - C6,0] ,[0,C1 - C2 + C5 - C6,C1 - C2 + C5 - C6,0] , [0,0,0,2 * (C1 - C2  + C5 - C6 )]] )
(Meigvec, Meigvals) = NNJ0.diagonalize(normalize=True)

print("NN States with J=0")
for i in range(len(NNstates)):
    print("Eigenvalue = ", Meigvals[i,i], "State = ",Meigvec.col(i).dot(NNstates))

NNJ1 = Matrix( [ [0,0,0,0] , [0,C1 + C2 + C5 + C6, -C1 - C2 - C5 - C6,0] ,[0,-C1 - C2 - C5 - C6,C1 + C2 + C5 + C6,0] , [0,0,0,0]] )
(Meigvec, Meigvals) = NNJ1.diagonalize(normalize=True)

print("NN States with J=1")
for i in range(len(NNstates)):
    print("Eigenvalue = ", Meigvals[i,i], "State = ",Meigvec.col(i).dot(NNstates))

#All Xi-N states

x0p = Symbol('Ξ⁰p')
x0n = Symbol('Ξ⁰n')
xmp = Symbol('Ξ⁻p')
xmn = Symbol('Ξ⁻n')
XNstates = Matrix( [x0p, x0n, xmp, xmn])

XNJ0 = Matrix([[-C3 + C4 + 2*C5 - 2*C6, 0, 0, 0], [0, -2*(C3 - C4 - C5 + C6), -C3 + C4, 0], [0, -C3 + C4, -2*(C3 - C4 - C5 + C6), 0], [0, 0, 0, -C3 + C4 + 2*C5 - 2*C6]])
(Meigvec, Meigvals) = XNJ0.diagonalize(normalize=True)

print("XiN States with J=0")
for i in range(len(XNstates)):
    print("Eigenvalue = ", Meigvals[i,i], "State = ",Meigvec.col(i).dot(XNstates))


#All Sigma-Sigma states

spsp = Symbol('Σ⁺Σ⁺')
sps0 = Symbol('Σ⁺Σ⁰')
spsm = Symbol('Σ⁺Σ⁻')
s0sp = Symbol('Σ⁰Σ⁺')
s0s0 = Symbol('Σ⁰Σ⁰')
s0sm = Symbol('Σ⁰Σ⁻')
smsp = Symbol('Σ⁻Σ⁺')
sms0 = Symbol('Σ⁻Σ⁰')
smsm = Symbol('Σ⁻Σ⁻')
SSstates = Matrix( [spsp, sps0, spsm, s0sp, s0s0, s0sm, smsp, sms0, smsm])

SSJ0 = Matrix([[2*(C1 - C2 + C5 - C6), 0, 0, 0, 0, 0, 0, 0, 0], [0, C1 - C2 + C5 - C6, 0, C1 - C2 + C5 - C6, 0, 0, 0, 0, 0], [0, 0, -C3 + C4 + C5 - C6, 0, -C1 + C2 - C3 + C4, 0, -C3 + C4 + C5 - C6, 0, 0], [0, C1 - C2 + C5 - C6, 0, C1 - C2 + C5 - C6, 0, 0, 0, 0, 0], [0, 0, -C1 + C2 - C3 + C4, 0, C1 - C2 - C3 + C4 + 2*C5 - 2*C6, 0, -C1 + C2 - C3 + C4, 0, 0], [0, 0, 0, 0, 0, C1 - C2 + C5 - C6, 0, C1 - C2 + C5 - C6, 0], [0, 0, -C3 + C4 + C5 - C6, 0, -C1 + C2 - C3 + C4, 0, -C3 + C4 + C5 - C6, 0, 0], [0, 0, 0, 0, 0, C1 - C2 + C5 - C6, 0, C1 - C2 + C5 - C6, 0], [0, 0, 0, 0, 0, 0, 0, 0, 2*(C1 - C2 + C5 - C6)]])
(Meigvec, Meigvals) = SSJ0.diagonalize(normalize=True)

print("SS States with J=0")
for i in range(len(SSstates)):
    print("Eigenvalue = ", Meigvals[i,i], "State = ",Meigvec.col(i).dot(SSstates))



#Isospin 0 only states

ll = Symbol('ΛΛ')
xn = Symbol('(Ξ⁰n-Ξ⁻p)/sqrt(2)')
ss = Symbol('(Σ⁰Σ⁰+Σ⁺Σ⁻+Σ⁻Σ⁺)/sqrt(3)')
LXNSstates = Matrix( [ll, ss, xn])

LXNSJ0 = Matrix([[C1 - C2 - C3 + C4 + 2*C5 - 2*C6, (C1 - C2 - C3 + C4)/sqrt(3), (-4*C1 + 4*C2 - 5*C3 + 5*C4)/3], [(C1 - C2 - C3 + C4)/sqrt(3), -C1 + C2 - 3*C3 + 3*C4 + 2*C5 - 2*C6, sqrt(3)*(-C3 + C4)], [(-4*C1 + 4*C2 - 5*C3 + 5*C4)/3, sqrt(3)*(-C3 + C4), -3*C3 + 3*C4 + 2*C5 - 2*C6]])
(Meigvec, Meigvals) = LXNSJ0.diagonalize(normalize=True)

print("Isospin 0 only States with J=0")
for i in range(len(LXNSstates)):
    print("Eigenvalue = ", Meigvals[i,i], "State = ",Meigvec.col(i).dot(LXNSstates))
