![](https://raw.githubusercontent.com/rafafrdz/CC_P2/master/img/1.png)![](https://raw.githubusercontent.com/rafafrdz/CC_P2/master/img/2.png)

![](https://raw.githubusercontent.com/rafafrdz/CC_P2/master/img/3.png)

```PYTHON
# -*- coding: utf-8 -*-
import time
from numpy import *
from scipy.sparse.linalg import cgs,splu
from scipy.sparse import lil_matrix,identity,spdiags
from scipy.linalg import lu_factor,lu_solve,cho_factor,cho_solve
from matplotlib.pyplot import *


def f0(x):
    y=cos(x)*(1-cos(x))
    return y
def u0(x):
    return 0*x
def u1(x):
    y = 1 - 2*x/pi
    return y

stop = 10**(-7)

def EJERCICIO1(x0,xf,N,alfa,ua,ub,fuente,cond0,parada,eps):
    t1=time.time()
    N=int(N)
    x0=float(x0)
    xf=float(xf)
    dx=(xf-x0)/float(N)
    dx2=dx*dx
    x=linspace(x0,xf,N+1)
    M = lil_matrix((N+1,N+1), dtype='float64')
    Id=identity(N+1,dtype='float64',format='csc')

    M.setdiag(2.0*ones(N+1),0)
    M.setdiag(-1.0*ones(N+1),1)
    M.setdiag(-1.0*ones(N+1),-1)
    b=fuente(x)
    #M[0,0]=1.0
    M[0,1]=0.0
    #M[N,N]=1.0
    M[N,N-1]=0.0
    M[N-1,N]=0.0
    M[1,0]=0.0
    M=M.tocsc()
    A=alfa/(dx2)*M
    A[0,0]=1.0
    A[N,N]=1.0

    LU=splu(A)
    usol = cond0(x)
    i = 0
    error = 2*eps
    while((i<parada) & (error>eps)):
        b = fuente(x) + usol**2
        b[0]=ua
        b[N]=ub
        b[N-1]+=ub*alfa/dx2
        b[1]+=ua*alfa/dx2
        i += 1
        usol2 = copy(usol)
        usol=LU.solve(b)
        error = max(abs(usol-usol2))
        plot(x,usol,'b')

    tf=time.time()
    print "Tiempo de ejecucion:",tf-t1
    print "Numero de iteraciones:",i
    print "error:", error
    plot(x,usol,'b')
    plot(x,cos(x),'ro')
    show()

#EJERCICIO1(0,pi,50,1.0,1,-1,f0,u0,500,stop)
def EJERCICIO2(x0,xf,N,alfa,ua,ub,fuente,cond0,parada,eps):
    t1=time.time()
    N=int(N)
    x0=float(x0)
    xf=float(xf)
    dx=(xf-x0)/float(N)
    dx2=dx*dx
    x=linspace(x0,xf,N+1)
    usol = cond0(x)
    M = lil_matrix((N+1,N+1), dtype='float64')
    Id=identity(N+1,dtype='float64',format='csc')
    D = lil_matrix((N+1,N+1), dtype='float64')
    for i in range(N):
        D[i,i]=usol[i]

    M.setdiag(2.0*ones(N+1),0)
    M.setdiag(-1.0*ones(N+1),1)
    M.setdiag(-1.0*ones(N+1),-1)
    b=fuente(x)
    M[0,0]=1.0
    M[0,1]=0.0
    M[N,N]=1.0
    M[N,N-1]=0.0
    M[N-1,N]=0.0
    M[1,0]=0.0
    M=M.tocsc()
    A=alfa/(dx2)*M-D
    LU=splu(A)
    i = 0
    error = 2*eps
    #for i in range(20):
    while((i<parada) & (error>eps)):
        b = fuente(x) #+ usol**2
        b[0]=ua
        b[N]=ub
        b[N-1]+=ub*alfa/dx2
        b[1]+=ua*alfa/dx2
        D = lil_matrix((N+1,N+1), dtype='float64')
        for j in range(N):
            D[j,j]=usol[j]
        A=alfa/(dx2)*M-D
        A[0,0]=1.0
        A[N,N]=1.0
        LU=splu(A)
        i += 1
        usol2 = copy(usol)
        usol=LU.solve(b)
        error = max(abs(usol-usol2))
        plot(x,usol,'b')

    tf=time.time()
    print "Tiempo de ejecucion:",tf-t1
    print "Numero de iteraciones:",i
    print "error:", error
    plot(x,usol,'b')
    plot(x,cos(x),'ro')
    show()
#EJERCICIO2(0,pi,50,1.0,1,-1,f0,u0,500,stop)

def EJERCICIO3(x0,xf,N,T,Nt,alfa,ua,ub,fuente,cond0,parada,eps):
    t1=time.time()
    N=int(N)
    x0=float(x0)
    xf=float(xf)
    dx=(xf-x0)/float(N)
    dx2=dx*dx
    x=linspace(x0,xf,N+1)
    dt = T/float(Nt)
    usol = cond0(x)
    M = lil_matrix((N+1,N+1), dtype='float64')
    Id=identity(N+1,dtype='float64',format='csc')
    D = lil_matrix((N+1,N+1), dtype='float64')
    for i in range(N):
        D[i,i]=usol[i]

    M.setdiag(2.0*ones(N+1),0)
    M.setdiag(-1.0*ones(N+1),1)
    M.setdiag(-1.0*ones(N+1),-1)
    b=fuente(x)
    M[0,0]=0.0
    M[0,1]=0.0
    M[N,N]=0.0
    M[N,N-1]=0.0
    M[N-1,N]=0.0
    M[1,0]=0.0
    M=M.tocsc()
    A=Id + alfa*dt/(dx2)*M
    LU=splu(A)
    i = 0
    error = 2*eps
    #for i in range(20):
    while((i<parada) & (error>eps)):
        b = dt*(fuente(x) + usol**2) + usol
        b[0]=ua
        b[N]=ub
        b[N-1]+=ub*alfa*dt/dx2
        b[1]+=ua*alfa*dt/dx2
        i += 1
        usol2 = copy(usol)
        usol=LU.solve(b)
        error = max(abs(usol-usol2))
        plot(x,usol,'b')

    tf=time.time()
    print "Tiempo de ejecucion:",tf-t1
    print "Numero de iteraciones:",i
    print "error:", error
    plot(x,usol,'b')
    plot(x,cos(x),'r')
    show()
#EJERCICIO3(0,pi,50,1,30,1.0,1,-1,f0,u1,500,stop)

```

