import numpy as np
from models import *

X = np.array([2,3,4,7,3,9,8])
Y = np.array([2,6,8,10,1,9,3])

X1,Y1 = None,None
with open('vstupy.txt','r') as vstupy:
    X1 = np.array(vstupy.readlines())
    X1 = np.array(list(map(lambda x : list(map(lambda number : int(number),x.split())),X1)))

with open('vystupy.txt','r') as vstupy:
    Y1 = np.array(vstupy.readlines())
    Y1 = np.array(list(map(lambda x : list(map(lambda number : int(number),x.split())),Y1)))

X1scaled = X1/X1.max()
Y1scaled = Y1/Y1.max()


print('additive model:')
ad = AdditiveModel(X1,Y1)
ad.solve()

print('additive model scaled data :')
ad = AdditiveModel(X1scaled,Y1scaled)
ad.solve()

print('additive model:')
ad = AdditiveModel(X,Y)
ad.solve()

# print('weighted additive model:')
# wad = WeightedAdditiveModel(X1,Y1,np.ones(3),np.ones(2))
# wad.solve()

# print('weighted additive model:')
# wad = WeightedAdditiveModel(X1,Y1,np.ones(3),np.ones(2))
# wad.solve()

# print('mep model:')
# mep = MeasureEfficiencyProportionModel(X1,Y1)
# mep.solve()

# print('mep model scaled Data:')
# mep = MeasureEfficiencyProportionModel(X1scaled,Y1scaled)
# mep.solve()

# print('mep model:')
# mep = MeasureEfficiencyProportionModel(X,Y)
# mep.solve()

# print('bam model:')
# bam = BoundedAdjustedMeasureModel(X1,Y1)
# bam.solve()

# print('bam model scaled data:')
# bam = BoundedAdjustedMeasureModel(X1scaled,Y1scaled)
# bam.solve()

# print('bam model:')
# bam = BoundedAdjustedMeasureModel(X,Y)
# bam.solve()

# print('russel Model')
# rm = RusselModel(X1,Y1)
# rm.solve()

# print('russel Model scaled data')
# rm = RusselModel(X1scaled,Y1scaled)
# rm.solve()

# print('russel Model')
# rm = RusselModel(X,Y)
# rm.solve()

# print('sbm')
# sbm = SlackedBasedMeausereModel(X1,Y1)
# sbm.solve()

# print('sbm')
# sbm = SlackedBasedMeausereModel(X,Y)
# sbm.solve()