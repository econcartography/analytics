# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 18:14:36 2015

@author: emanuele
"""
import numpy as np
import warnings

def fitnessComplexityOneStep(matrix,FitnessI):
    #One step of the algorithm
    ComplexityTemp=np.zeros(np.shape(matrix)[1])
    FitnessTemp=np.zeros(np.shape(matrix)[0])
    for p in range(np.shape(matrix)[1]):
        try:
            sumCT=matrix[:,p]/FitnessI
            sumCT[np.where(np.isnan(sumCT))]=0.
            sumCT=sumCT*(1-np.isnan(sumCT))
            ComplexityTemp[p] = 1./np.sum(sumCT)
            if np.isnan(ComplexityTemp[p]):
                ComplexityTemp[p]=0
        except ZeroDivisionError:
            ComplexityTemp[p] = 0
    Complexity=ComplexityTemp/np.mean(ComplexityTemp)
    FitnessTemp = np.dot(matrix, Complexity)
    Fitness=FitnessTemp/np.mean(FitnessTemp)
    return Fitness,Complexity

def fitnessComplexityStop(matrix,iteraz,verbose=0,stop=100000,Fitness=[0],Complexity=[0],stopCondition=0):
    #Fitness-Complexity algorithm with automatic stop (see Pugliese, Zaccaria, Pietronero, 2015, arXiv)
    #Stop condition:0 - when countries' MCI is above stop
    #               1 - when both countries' and products' MCI is above stop
    #               2 - when products' MCI is above stop
    if len(Fitness)==1:
        if hasattr(matrix,'getformat'):
            Fitness=[np.array([country.sum()*matrix.shape[0]/matrix.sum() for country in matrix]) for i in range(0,3)]
        else:
            Fitness=[np.array([np.sum(country)*len(matrix)/np.sum(matrix) for country in matrix]) for i in range(0,3)]
    else:
        Fitness=[np.array(Fitness) for i in range(0,3)]
    if len(Complexity)==1:
        Complexity=[np.array([1. for sector in matrix[0]]) for i in range(0,3)]
    else:
        Complexity=[np.array(Complexity) for i in range(0,3)]
    OrderC=[len(Fitness[0]) for c in Fitness[0]]
    OrderP=[len(Complexity[0]) for p in Complexity[0]]
    deadC=0
    deadP=0
    deadCTot=[]
    minCrossingArray=[]
    for i in range(0,iteraz):
        if i%50 ==0 : print (i),
        Fitness[i%3],Complexity[i%3]=fitnessComplexityOneStep(matrix,Fitness[(i-1)%3])
        #print Fitness

        deads=[(country[1],country[2]) for country in zip(Fitness[i%3],Fitness[(i-1)%3],range(0,len(Fitness[0]))) 
            if country[0]==0 and not country[1]==0]
        if len(deads)>0:
            deads.sort()
            for dead in deads:
                OrderC[dead[1]]=deadC
                deadC=deadC+1
            deadCTot=deadCTot+[dead[1] for dead in deads]
        deads=[(product[1],product[2]) for product in zip(Complexity[i%3],Complexity[(i-1)%3],range(0,len(Complexity[0]))) 
            if product[0]==0 and not product[1]==0]
        if len(deads)>0:
            deads.sort()
            for dead in deads:
                OrderP[dead[1]]=deadP
                deadP=deadP+1
        if i > 10:
            gc=(np.log(Fitness[i%3])-np.log(Fitness[(i-1)%3]))/np.log(1.0*i/(i-1))
            gp=(np.log(Complexity[i%3])-np.log(Complexity[(i-1)%3]))/np.log(1.0*i/(i-1))
            minCrossingC=MinCrossingTime(Fitness[i%3],gc,i,stop)
            minCrossingP=MinCrossingTime(Complexity[i%3],gp,i,stop)
            if verbose==2 or (verbose==1 and i%10==0):
                print (i), minCrossingC,minCrossingP
            minCrossingArray.append(minCrossingC)
            if stopCondition==0:
                if minCrossingC==stop:
                    break
            if stopCondition==1:
                if minCrossingC==stop and minCrossingP==stop:
                    break
            if stopCondition==2:
                if minCrossingP==stop:
                    break
    sortedFit=sorted([ (country[0],country[1]) 
        for country in zip(Fitness[i%3],range(0,len(Fitness[0]))) 
        if country[0]>0 ])
    for country in sortedFit:
        OrderC[country[1]]=deadC
        deadC=deadC+1
    sortedCom=sorted([ (product[0],product[1]) 
      for product in zip(Complexity[i%3],range(0,len(Complexity[0]))) 
      if product[0]>0 ])
    for product in sortedCom:
        OrderP[product[1]]=deadP
        deadP=deadP+1
       
    return Fitness[i%3],Complexity[i%3],gc,gp,OrderC,OrderP

def MinCrossingTime(variables,growthrates,iteraz,maxN):
    #Minimum time before the next crossing
    rank=zip(variables,growthrates,range(0,len(variables)))
    rank=[element for element in rank if not rank[0]==0]
    rank.sort()
    CrossingTimes=[np.power(rank[i][0]/rank[i+1][0]*np.power(iteraz,rank[i+1][1]-rank[i][1]),1./(rank[i+1][1]-rank[i][1])) 
    if not rank[i+1][1]>=rank[i][1] else maxN 
    for i in range(0,len(rank)-1)]
    
    return min([ct for ct in CrossingTimes if ct>0]+[maxN])

