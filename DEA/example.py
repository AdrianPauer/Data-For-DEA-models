from models import *
ad = RusselModel("testData/simpleX.txt","testData/simpleY.txt",vrs=True)#,nonDiscretionaryInputs=np.array([0]))
ad.solve()
print(ad)
ad.plot_projections()
ad.plot_effectivity()

# ad = AdditiveModel("testData/simpleX.txt","testData/simpleY.txt",vrs=False)
# ad.solve()
# print(ad)
# ad.plot_projections()
# ad.plot_effectivity()

# ad = AdditiveModel("testData/simpleX.txt","testData/simpleY.txt",vrs=True)
# ad.solve()
# print(ad)
# ad.plot_projections()
# ad.plot_effectivity()

# ad = AdditiveModel("testData/simpleX.txt","testData/simpleY.txt",vrs=True,nonDiscretionaryInputs=np.array([0]))
# ad.solve()
# print(ad)
# ad.plot_projections()
# ad.plot_effectivity()
