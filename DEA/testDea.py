import numpy as np
from models import *

X = np.array([2,3,4,7,3,9,8])
Y = np.array([2,6,8,10,1,9,3])

def getDataFromFile(inputFile,outputFile):
    with open(inputFile,'r') as vstupy:
        X = np.array(vstupy.readlines())
        X = np.array(list(map(lambda x : list(map(lambda number : float(number),x.split())),X)))

    with open(outputFile,'r') as vstupy:
        Y = np.array(vstupy.readlines())
        Y = np.array(list(map(lambda x : list(map(lambda number : float(number),x.split())),Y)))
    return X,Y

X1,Y1= getDataFromFile("testData/vstupy.txt","testData/vystupy.txt")
#XLessThatZero,YLessThatZero = getDataFromFile("testData/vstupyLessThanZero.txt","testData/vystupyLessThanZero.txt")
#X2,YIncomplete  =  getDataFromFile("testData/vstupy.txt","testData/vystupyMissing.txt")

#print(type(YIncomplete[0]))

# print(getDataFromFile("testData/simpleX.txt","testData/simpleY.txt")[0])
# print('additive model:')
# ad = AdditiveModel("testData/simpleX.txt","testData/simpleY.txt",vrs=False)
# ad.solve()
# print(ad)
# ad.plot_projections()
# ad.plot_effectivity()
#ad.plot_projections()
#ad.writeSolutionsToFile('ADoutput.txt')

# print('additive model scaled data with input 0 and output 0 restricted to be 0:')
# ad = AdditiveModel(X1scaled,Y1scaled,nonDiscretionaryInputs=np.array([0]), nonDiscretionaryOutputs=np.array([0]))
# ad.solve()

# print('additive model:')
# ad = AdditiveModel(X,Y,vrs = True)
# ad.solve()
# print(ad)
#print(*ad.get('projection'))

# print('additive model:')
# ad = AdditiveModel(X,Y,vrs = False,nonDiscretionaryInputs=np.array([0]),nonDiscretionaryOutputs=np.array([0]))
# ad.solve()


# print('weighted additive model:')
# wad = WeightedAdditiveModel(X1scaled,Y1scaled,np.ones(3),np.ones(2))
# wad.solve()

# print('weighted additive model scaled data :')
# wad = WeightedAdditiveModel(X1scaled,Y1scaled,np.ones(3),np.ones(2),nonDiscretionaryInputs=np.array([0,1]),nonDiscretionaryOutputs=np.array([0]))
# wad.solve()


# print('weighted additive model:')
# wad = AdditiveModel(X,Y,vrs = True)
# wad.solve()

# print('weighted additive model:')
# wad = AdditiveModel(X,Y,vrs = True,nonDiscretionaryInputs=np.array([0]),nonDiscretionaryOutputs=np.array([0]))
# wad.solve()


# print('mep model scaled Data:')
# mep = MeasureEfficiencyProportionModel(X1scaled,Y1scaled)
# mep.solve()


# print('mep model scaled Data:')
# mep = MeasureEfficiencyProportionModel(X1scaled,Y1scaled,nonDiscretionaryInputs=np.array([0]),nonDiscretionaryOutputs=np.array([0]))
# mep.solve()


# print('mep model:')
# mep = MeasureEfficiencyProportionModel(X,Y,vrs = True)
# mep.solve()

# print('bam model:')
# bam = BoundedAdjustedMeasureModel(X,Y,vrs=True)
# bam.solve()
# print(np.apply_along_axis(np.sum, 1,bam.get('lambda')))
# print(bam)

# print('bam model scaled data:')
# bam = BoundedAdjustedMeasureModel(X1scaled,Y1scaled, nonDiscretionaryInputs=np.array([0]),nonDiscretionaryOutputs=np.array([0]))
# bam.solve()

# print('bam model:')
# bam = BoundedAdjustedMeasureModel(X,Y,vrs = False)
# bam.solve()

# print('bam model:')
# bam = BoundedAdjustedMeasureModel(X,Y,vrs = True)
# bam.solve()

# print('russel Model')
# rm = RusselModel(X,Y,vrs = True)
# rm.solve()
# print(np.all(abs(np.apply_along_axis(np.sum, 1,rm.get('lambda')) - 1)  <= 0.0001))
# print(rm)

print('russel Model scaled data')
rm = RusselModel("testData/vstupy.txt","testData/vystupy.txt",vrs = True,nonDiscretionaryInputs=np.array([0]),nonDiscretionaryOutputs=np.array([0]))
rm.solve()
print(rm)

# print('russel Model')
# rm = RusselModel(X1,Y1,vrs=True)
# rm.solve()

# print('sbm')
# sbm = SlackedBasedMeausereModel(X1scaled, Y1scaled)
# sbm.solve()

# print('sbm')
# sbm = SlackedBasedMeausereModel(X1scaled, Y1scaled, nonDiscretionaryInputs=np.array([0,1]),nonDiscretionaryOutputs=np.array([0]))
# sbm.solve()

# print('sbm')
# sbm = SlackedBasedMeausereModel(X,Y,vrs = True,nonDiscretionaryOutputs=np.array([0]))
# sbm.solve()