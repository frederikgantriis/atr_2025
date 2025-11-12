import math
import typing

target = [0.5,0.5,0.5]


def growAndMeasure(individual,generation,indvNumber): #Call tile functions here
    
    print(generation,indvNumber)
    grownIndividual = []
    for i in range(len(individual)):       
      grownIndividual.append(individual[i])

    print("Evaluating: ",grownIndividual)
    #overallSum = sum(grownIndividual)
    
    measure = [grownIndividual[0],grownIndividual[1],grownIndividual[2]]#makeColor(grownIndividual[0],grownIndividual[1],grownIndividual[2])

    return measure

def evaluate(individual,generation,indvNumber):
      
      
      
      measure = growAndMeasure(individual,generation,indvNumber) #Call a function to start the tiles and return a sensor value 

      distances  = [10000] * len(target)

      for i in range(len(distances)):
        distances[i] = target[i]-measure[i]
        distances[i] = distances[i]*distances[i]

      errorSum = sum(distances)
      meanSquaredError = errorSum/(len(target))

      return meanSquaredError


