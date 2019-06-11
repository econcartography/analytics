'''Compute fitnes and complexity'''
import numpy

def computeFQ1(Mcp, nI= 100):
    Mpc = Mcp.transpose()
   
    numberCountries = len(Mcp)
    numberProducts = len(Mcp[0])

    fitnessArrayPrev = numpy.ones((numberCountries,1))
    fitnessArray = numpy.ones((numberCountries,1)) 
    auxFitnessArray = numpy.ones((numberCountries,1))
    #print fitness_array, aux_fitness_array 
   
    qualityArrayPrev = numpy.ones((numberProducts,1))
    qualityArray = numpy.ones((numberProducts,1)) 
    auxQualityArray = numpy.ones((numberProducts,1))
    auxUglinessArray = numpy.ones((numberProducts,1))
    #print quality_array, aux_quality_array 
   
    average = 0   
    for j in range(0,nI+1):
        fitnessArrayPrev = fitnessArray
        qualityArrayPrev = qualityArray

        auxFitnessArray = numpy.dot(Mcp,qualityArrayPrev)
        #print aux_fitness_array
        averageFitness = numpy.sum(auxFitnessArray)/numberCountries
      
        fitnessArray = auxFitnessArray/averageFitness
        #print fitness_array
        auxUglinessArray = numpy.dot(Mpc,1/fitnessArrayPrev)
        auxQualityArray = 1/auxUglinessArray
      
        averageQuality = numpy.sum(auxQualityArray)/numberProducts 
        qualityArray = auxQualityArray/averageQuality

    return numpy.array([f[0] for f in fitnessArray]),  numpy.array([c[0] for c in qualityArray]) 
