# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 11:30:53 2018

@author: BENDJOUDI
"""
import numpy as np
from numpy import random as rd
# =============================================================================
#                       Create FLP Model Function
# =============================================================================

def CreateModel():
#     Size of Blocks
    w=[572,572,572,572,88.9,572,572,333,80,80]   # Width
    h=[82.5,82.5,82.5,82.5,84,82.5,82.5,84,45,45]   #Height
    
    delta=[20,20,20,20,40,20,20,20,40,40]     #Gap
    
    rin  = [0.7,0.7,0.7,0.7,0.04,0.7,0.7,0.7,0.2,0.1] #Location of Input Gate
    rout = [0.03,0.02,0.03,0.03,0.7,0.07,0.07,0.03,0.7,0.7] #Location of Output Gate
    
    n= np.shape(w)[0]
    a=[[ 0,5,1,0,0,0,0,1,0,0],
       [ 1,0,5,0,0,0,0,1,0,0],
       [ 0,1,0,1,0,0,0,1,0,0],
       [ 0,0,5,0,0,5,1,1,0,0],
       [ 0,0,0,0,0,0,0,0,1,1],
       [ 0,0,0,1,0,0,5,1,0,0],
       [ 0,0,0,1,0,5,0,1,0,0],
       [ 1,1,1,1,0,1,1,0,0,0],
       [ 0,0,0,0,1,0,1,0,0,1],
       [ 0,0,0,0,1,0,0,0,1,0]]

    
    W=1000
    H=800
    
    phi=50000
    xin=np.zeros((n,))
    yin=np.zeros((n,))
    
    xout=np.zeros((n,))
    yout=np.zeros((n,))

    for i in range(0,n):
        if ( rin[i]>=0 ) and ( rin[i]<= 0.25):
            xin[i] = (4*rin[i]-0.5)*w[i]
            yin[i] = -h[i]/2
            
        if ( rin[i]>0.25 ) and ( rin[i] <=0.5 ):
            xin[i]= w[i]/2
            yin[i]= (4*rin[i]-1.5)*h[i]
            
        if rin[i]>0.5 and rin[i]<=0.75:
            xin[i]= (2.5-4*rin[i])*w[i]
            yin[i] = h[i]/2
            
        if rin[i]>0.75 and rin[i]<=1:
            xin[i] = -w[i]/2
            yin[i] = (3.5-4*rin[i])*h[i]
            
        
        
        if rout[i]>=0 and rout[i]<=0.25:
            xout[i] = (4*rout[i]-0.5)*w[i]
            yout[i] = -h[i]/2 
            
        if rout[i]>0.25 and rout[i]<=0.5:
            xout[i] = w[i]/2
            yout[i] = (4*rout[i]-1.5)*h[i] 
            
        if rout[i]>0.5 and rout[i]<=0.75:
            xout[i] = (2.5-4*rout[i])*w[i]
            yout[i] = h[i]/2 
            
        if rout[i]>0.75 and rout[i]<=1:
            xout[i] = -w[i]/2
            yout[i] = (3.5-4*rout[i])*h[i]

    
    
    model = {'n':n,
            'w':w,
            'h':h,
            'delta':delta,
            'rin':rin,
            'xin':xin,
            'yin':yin,
            'rout':rout,
            'xout':xout,
            'yout':yout,
            'a':a,
            'W':W,
            'H':H,
            'phi':phi
            }
    
    return model


# =============================================================================
#                       Set Decision Vars Function
# =============================================================================

def SetVars(model):
    
    xhat = {'Min':0,
            'Max':1,
            'Size':[1,model['n']] } 
    xhat['Count'] = np.prod(xhat['Size'])
    xhat['VelMax'] = 0.1*(xhat['Max']-xhat['Min'])
    xhat['VelMin'] = -xhat['VelMax']
    
    
    
    yhat = {'Min':0,
            'Max':1,
            'Size':[1,model['n']] }
    yhat['Count'] = np.prod(yhat['Size'])
    yhat['VelMax'] = 0.1*(yhat['Max']-yhat['Min'])
    yhat['VelMin'] = -yhat['VelMax']
            
    
    
    rhat = {'Min':0,
            'Max':1,
            'Size':[1,model['n']] }
    rhat['Count'] = np.prod(rhat['Size'])
    rhat['VelMax'] = 0.1*(rhat['Max']-rhat['Min'])
    rhat['VelMin'] = -rhat['VelMax']
    
    
    Vars = { 'xhat': xhat,
             'yhat': yhat,
             'rhat': rhat }
    
    return Vars


# =============================================================================
#           Create random solution for initialize Decision Vars
# =============================================================================
def CreateRandomSolution(model):
    sol= {'xhat': np.random.rand(1,model['n']).reshape((model['n'],)),
          'yhat': np.random.rand(1,model['n']).reshape((model['n'],)),
          'rhat': np.random.rand(1,model['n']).reshape((model['n'],))
        }
    return sol
# =============================================================================
#               Cost function  
# =============================================================================

def ParseSolution(sol1,model):
    n= model['n']
    delta=model['delta']
    W=model['W']
    H=model['H']
    a=model['a']

 
    rhat=sol1['rhat']
    r=np.minimum(np.floor(4*rhat),3)
    theta=r*np.pi/2
    
    w = abs(np.cos(theta))*model['w'] + abs(np.sin(theta))* model['h']

    w = w.reshape((n,))
    
    
    h = abs(np.cos(theta)) * model['h'] +  abs(np.sin(theta))* model['w']
    h = h.reshape((n,))


    xhat=sol1['xhat']
    xmin=w/2+delta
    xmax=W-xmin
    x=xmin+(xmax-xmin)*xhat
    x = x.reshape((n,))

    
    yhat=sol1['yhat']
    ymin=h/2+delta
    ymax=H-ymin
    y=ymin+(ymax-ymin)*yhat
    y = y.reshape((n,))

    xl=x-w/2
    yl=y-h/2
    xu=x+w/2
    yu=y+h/2
    
    xin=x+np.cos(theta)*model['xin']-np.sin(theta)*model['yin']
    xin = xin.reshape((n,))

 
    yin=y+np.cos(theta)*model['yin']+np.sin(theta)*model['xin']
    yin = yin.reshape((n,))    

    
    xout=x+np.cos(theta)*model['xout']-np.sin(theta)*model['yout']
    xout = xout.reshape((n,))


    yout=y+np.cos(theta)*model['yout']+np.sin(theta)*model['xout']
    yout = yout.reshape((n,))
   


    d=np.zeros((n,n))
    V=np.zeros((n,n))
    for i in range(0,n):
        for j in range(i+1,n):
            d[i][j]=np.linalg.norm([(xout[i]-xin[j]),(yout[i]-yin[j])])
            d[j][i]=np.linalg.norm([(xout[j]-xin[i]),(yout[j]-yin[i])])

            
            DELTA=np.maximum(delta[i],delta[j])
            XVij=max(0,1-abs(x[i]-x[j])/((w[i]+w[j])/2+DELTA))
            YVij=max(0,1-abs(y[i]-y[j])/((h[i]+h[j])/2+DELTA))
            
            V[i][j]=min(XVij,YVij)
            V[j][i]=V[i,j]

    
    v=np.mean(V[:])
    
    ad=a*d

    
    SumAD=np.sum(ad)
    
    XMIN=min(xl)
    YMIN=min(yl)
    XMAX=max(xu)
    YMAX=max(yu)
    
    ContainerArea=(XMAX-XMIN)*(YMAX-YMIN)
    MachinesArea=np.sum(w*h)
    UnusedArea=np.maximum(ContainerArea-MachinesArea,0)
    UnusedAreaRatio=UnusedArea/ContainerArea
    UnusedAreaCost=model['phi']*UnusedAreaRatio
    
    #beta=100;
    #z=SumAD*(1+beta*v);
    
    alpha=1e12
    z=SumAD+UnusedAreaCost+alpha*v


    sol2 = {'r':r,
           'theta':theta,
           'x':x,
           'y':y,
           'xl':xl,
           'yl':yl,
           'xu':xu,
           'yu':yu,
           'xin':xin,
           'yin':yin,
           'xout':xout,
           'yout':yout,
           'd':d,
           'ad':ad,
           'SumAD':SumAD,
           'XMIN':XMIN,
           'YMIN':YMIN,
           'XMAX':XMAX,
           'YMAX':YMAX,
           'ContainerArea':ContainerArea,
           'MachinesArea':MachinesArea,
           'UnusedArea':UnusedArea,
           'UnusedAreaRatio':UnusedAreaRatio,
           'UnusedAreaCost':UnusedAreaCost,
           'V':V,
           'v':v,
           'IsFeasible':(v==0),
           'z':z
                 }
    return sol2


def CostFunction(sol1,model):
     
    sol = ParseSolution(sol1,model)
    cost = sol['z']
    return cost,sol

# =============================================================================
#                   Mutation Function
# =============================================================================

def Mutate(sol,Vars):

    newsol=sol

    sigma=0.1*(Vars['xhat']['Max']-Vars['xhat']['Min']);
    j=rd.randint(1,Vars['xhat']['Count'])
    dxhat=sigma*rd.randn()
    newsol['xhat'][j]=sol['xhat'][j]+dxhat
    for i in range(len(newsol['xhat'])):
        newsol['xhat'][i]=max(newsol['xhat'][i],Vars['xhat']['Min'])
        newsol['xhat'][i]=min(newsol['xhat'][i],Vars['xhat']['Max'])
    
    sigma=0.1*(Vars['yhat']['Max']-Vars['yhat']['Min'])
    j=rd.randint(1,Vars['yhat']['Count'])
    dyhat=sigma*rd.randn()
    newsol['yhat'][j]=sol['yhat'][j]+dyhat
    for i in range(len(newsol['yhat'])):
        newsol['yhat'][i]=max(newsol['yhat'][i],Vars['yhat']['Min'])
        newsol['yhat'][i]=min(newsol['yhat'][i],Vars['yhat']['Max'])

    return newsol


# =============================================================================
#           Improve solution --- Local search methode
# =============================================================================

def ImproveSolution(sol1,model,Vars):

    n=model['n']

    A=rd.permutation(n)
    
    for i in A:
        sol1= MoveMachine(i,sol1,model,Vars)  

        return sol1

def RotateMachine(i,sol1,model):


    newsol0 = sol1.copy()
    newsol0['rhat'][i]=0.1
    cos,sol = CostFunction(newsol0,model)
    Newsol = [newsol0]
    z = [cos]
    
    newsol1 = sol1.copy()
    newsol1['rhat'][i]=0.35
    cos,sol = CostFunction(newsol1,model)
    Newsol.append(newsol1)
    z.append(cos)
    
    newsol2 = sol1.copy()
    newsol2['rhat'][i]=0.65
    cos,sol = CostFunction(newsol2,model)
    Newsol.append(newsol2)
    z.append(cos)
    
    newsol3 = sol1.copy()
    newsol3['rhat'][i] = 0.85
    cos,sol = CostFunction(newsol3,model)
    Newsol.append(newsol3)
    z.append(cos)


    
    z2 = min(z)
    ind = z.index(z2)
    sol2= Newsol[ind]
    return sol2,z2


def MoveMachine(i,sol1,model,Vars):

    dmax=0.5;
    # Zero

    Newsol0, z0=RotateMachine(i,sol1,model)
    z = [z0]
    Newsol= [Newsol0]
    
    # Move Up
    newsol2 = sol1.copy()
    dy=rd.uniform(0,dmax)
    newsol2['yhat'][i]=sol1['yhat'][i]+dy
    newsol2['yhat'][i]=max(newsol2['yhat'][i],Vars['yhat']['Min'])
    newsol2['yhat'][i]=min(newsol2['yhat'][i],Vars['yhat']['Max'])

    Newsol2, z2 = RotateMachine(i,newsol2,model)
    z.append(z2)
    Newsol.append(Newsol2)
#    newsol.append(Newsol2)
    
    # Move Down
    newsol3=sol1.copy()
    dy=rd.uniform(0,dmax)
    newsol3['yhat'][i]=sol1['yhat'][i]-dy
    newsol3['yhat'][i]=max(newsol3['yhat'][i],Vars['yhat']['Min'])
    newsol3['yhat'][i]=min(newsol3['yhat'][i],Vars['yhat']['Max'])
    


    Newsol3, z3 = RotateMachine(i,newsol3,model)
    z.append(z3)
    Newsol.append(Newsol3)
    
    # Move Right
    newsol4=sol1.copy()
    dx=rd.uniform(0,dmax)
    newsol4['xhat'][i]=sol1['xhat'][i]+dx
    newsol4['xhat'][i]=max(newsol4['xhat'][i],Vars['xhat']['Min'])
    newsol4['xhat'][i]=min(newsol4['xhat'][i],Vars['xhat']['Max'])
    


    Newsol4, z4 = RotateMachine(i,newsol4,model)
    z.append(z4)
    Newsol.append(Newsol4)
    
    # Move Left
    newsol5=sol1.copy()
    dx=rd.uniform(0,dmax)
    newsol5['xhat'][i]=sol1['xhat'][i]-dx;
    newsol5['xhat'][i]=max(newsol5['xhat'][i],Vars['xhat']['Min'])
    newsol5['xhat'][i]=min(newsol5['xhat'][i],Vars['xhat']['Max'])



    Newsol5, z5 = RotateMachine(i,newsol5,model)
    z.append(z5)
    Newsol.append(Newsol5)

    # Move Up-Right
    newsol6=sol1.copy()
    dx=rd.uniform(0,dmax)
    newsol6['xhat'][i]=sol1['xhat'][i]+dx
    newsol6['xhat'][i]=max(newsol6['xhat'][i],Vars['xhat']['Min'])
    newsol6['xhat'][i]=min(newsol6['xhat'][i],Vars['xhat']['Max'])
    dy=rd.uniform(0,dmax)
    newsol6['yhat'][i]=sol1['yhat'][i]+dy
    newsol6['yhat'][i]=max(newsol6['yhat'][i],Vars['yhat']['Min'])
    newsol6['yhat'][i]=min(newsol6['yhat'][i],Vars['yhat']['Max'])


 
    Newsol6, z6 = RotateMachine(i,newsol6,model)
    z.append(z6)
    Newsol.append(Newsol6)

    # Move Up-Left
    newsol7=sol1.copy()
    dx=rd.uniform(0,dmax)
    newsol7['xhat'][i]=sol1['xhat'][i]-dx
    newsol7['xhat'][i]=max(newsol7['xhat'][i],Vars['xhat']['Min'])
    newsol7['xhat'][i]=min(newsol7['xhat'][i],Vars['xhat']['Max'])
    dy=rd.uniform(0,dmax)
    newsol7['yhat'][i]=sol1['yhat'][i]+dy
    newsol7['yhat'][i]=max(newsol7['yhat'][i],Vars['yhat']['Min'])
    newsol7['yhat'][i]=min(newsol7['yhat'][i],Vars['yhat']['Max'])


    Newsol7, z7 = RotateMachine(i,newsol7,model)
    z.append(z7)
    Newsol.append(Newsol7)


    # Move Down-Right
    newsol8=sol1.copy()
    dx=rd.uniform(0,dmax)
    newsol8['xhat'][i]=sol1['xhat'][i]+dx
    newsol8['xhat'][i]=max(newsol8['xhat'][i],Vars['xhat']['Min'])
    newsol8['xhat'][i]=min(newsol8['xhat'][i],Vars['xhat']['Max'])
    dy=rd.uniform(0,dmax)
    newsol8['yhat'][i]=sol1['yhat'][i]-dy
    newsol8['yhat'][i]=max(newsol8['yhat'][i],Vars['yhat']['Min'])
    newsol8['yhat'][i]=min(newsol8['yhat'][i],Vars['yhat']['Max'])



    Newsol8, z8 = RotateMachine(i,newsol8,model)
    z.append(z8)
    Newsol.append(Newsol8)

    # Move Down-Left
    newsol9=sol1.copy()
    dx=rd.uniform(0,dmax)
    newsol9['xhat'][i]=sol1['xhat'][i] - dx
    newsol9['xhat'][i]=max(newsol9['xhat'][i],Vars['xhat']['Min'])
    newsol9['xhat'][i]=min(newsol9['xhat'][i],Vars['xhat']['Max'])
    dy=rd.uniform(0,dmax)
    newsol9['yhat'][i]=sol1['yhat'][i] - dy
    newsol9['yhat'][i]=max(newsol9['yhat'][i],Vars['yhat']['Min'])
    newsol9['yhat'][i]=min(newsol9['yhat'][i],Vars['yhat']['Max'])
    


    Newsol9, z9 = RotateMachine(i,newsol9,model)
    z.append(z9)
    Newsol.append(Newsol9)
    
    Z2 = min(z)
    ind = z.index(Z2)
    print(ind)

    
    Sol=Newsol[ind]
    return Sol




# =============================================================================
#                   Inialize Population (swarm)
# =============================================================================

def Init(PopSize):
    model = CreateModel()
    Vars = SetVars(model)
    # Empty Particle Template
    empty_particle = {
        'position': None,
        'velocity': None,
        'cost': None,
        'best': {'position':None,'cost':np.inf,'sol':None},
        'sol': None }
    
    
    # Initialize Global Best
    gbest = {'position': {}, 'cost': np.inf , 'sol' : {}}
    
    # Create Initial Population
#    pop = np.array([empty_particle]*PopSize)
    pop = []
    for i in range(0, PopSize):
        pop.append(empty_particle.copy())   

        # Initialize position
        pop[i]['position'] = CreateRandomSolution(model)
        # Initialise Velocity
        pop[i]['velocity'] = {'xhat' : np.zeros((Vars['xhat']['Size'][0],Vars['xhat']['Size'][1])).reshape((model['n'],)),
                              'yhat': np.zeros((Vars['yhat']['Size'][0],Vars['yhat']['Size'][1])).reshape((model['n'],)),
                              'rhat': np.zeros((Vars['rhat']['Size'][0],Vars['rhat']['Size'][1])).reshape((model['n'],)) }
        
        
        pop[i]['cost'], pop[i]['sol'] = CostFunction(pop[i]['position'],model)
        pop[i]['best']['position'] = pop[i]['position'].copy()
        pop[i]['best']['cost'] = pop[i]['cost'].copy()
        pop[i]['best']['sol'] = pop[i]['sol'].copy()
        # Update best personnal
        if pop[i]['best']['cost'] < gbest['cost']:
            gbest['position'] = pop[i]['best']['position'].copy()
            gbest['cost'] = pop[i]['best']['cost'].copy()
            gbest['sol'] = pop[i]['best']['sol'].copy()

            
    return pop,gbest,model,Vars
