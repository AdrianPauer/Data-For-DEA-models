import unittest
import numpy as np
import sys
import os
# Get the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

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

X1,Y1 = getDataFromFile("testData/vstupy.txt","testData/vystupy.txt")
XI,YIncomplete  =  getDataFromFile("testData/vstupy.txt","testData/vystupyMissing.txt")
XLessThatZero,YLessThatZero = getDataFromFile("testData/vstupyLessThanZero.txt","testData/vystupyLessThanZero.txt")


class TestShapeOfData(unittest.TestCase):

    def test_wrong_shape_of_inputData(self):
        """
        output matrix with Missing values for some outputs
        """
        with self.assertRaises(ValueError):
            AdditiveModel(X,YIncomplete)
        with self.assertRaises(ValueError):
            RusselModel(X,YIncomplete)


    def test_good_shape_of_input_data(self):
        """
        model solve function should run sucessfully for data with good shape
        """
        mep,rm,sbm = MeasureEfficiencyProportionModel(X,Y),RusselModel(X,Y),SlackedBasedMeausereModel(X,Y)
        self.assertTrue(rm.solve())
        self.assertTrue(mep.solve())
        self.assertTrue(sbm.solve())

        mep,rm,sbm = MeasureEfficiencyProportionModel(X1,Y1),RusselModel(X1,Y1),SlackedBasedMeausereModel(X1,Y1)
        self.assertTrue(rm.solve())
        self.assertTrue(mep.solve())
        self.assertTrue(sbm.solve())

    def test_below_zero_data(self):
        '''
        data less than zero 
        '''
        with self.assertRaises(ValueError):
            AdditiveModel(XLessThatZero,YLessThatZero)
        with self.assertRaises(ValueError):
            RusselModel(XLessThatZero,YLessThatZero)

    def test_NR_index_out_of_range(self):
        '''
        vector for nondiscretionary inputs/oputputs sholud be in range [0,m-1],[0,s-1]
        '''
        with self.assertRaises(IndexError):
            AdditiveModel(X,Y,vrs = True,nonDiscretionaryInputs=np.array([10]),nonDiscretionaryOutputs=np.array([0]))
        with self.assertRaises(IndexError):
            AdditiveModel(X,Y,vrs = True,nonDiscretionaryInputs=np.array([0]),nonDiscretionaryOutputs=np.array([10]))

    # def test_wrong_type_of_input_variables(self):
    #     ''''
    #     Nondiscretionary vectors should be of type ndArray
    #     get function should get appropriate key 
    #     '''

    

class TestFeaturesOfModel(unittest.TestCase):
    def test_variable_returns_to_scale(self):
        '''
        elements of optimal lambda should sum to 1
        '''
        mep,rm,sbm,bam = MeasureEfficiencyProportionModel(X,Y,vrs=True),RusselModel(X,Y,vrs=True),SlackedBasedMeausereModel(X,Y,vrs=True),BoundedAdjustedMeasureModel(X,Y,vrs=True)
        for model in mep,rm,sbm,bam: model.solve()
        for model in mep,rm,bam :
            self.assertTrue(np.all(abs(np.apply_along_axis(np.sum, 1,model.get('lambda')) - 1)  <= 0.0001))
        self.assertTrue(np.all((np.apply_along_axis(np.sum, 1,sbm.get('lambda')) - sbm.get('t')[:,0]) <= 0.0001))

        mep,rm,sbm,bam = MeasureEfficiencyProportionModel(X1,Y1,vrs=True),RusselModel(X1,Y1,vrs=True),SlackedBasedMeausereModel(X1,Y1,vrs=True),BoundedAdjustedMeasureModel(X1,Y1,vrs=True)
        for model in mep,rm,sbm,bam: model.solve()
        for model in mep,rm,bam :
            self.assertTrue(np.all(abs(np.apply_along_axis(np.sum, 1,model.get('lambda')) - 1)  <= 0.0001))
        self.assertTrue(np.all((np.apply_along_axis(np.sum, 1,sbm.get('lambda')) - sbm.get('t')[:,0]) <= 0.0001))
    
    def test_nondiscretionary_variables(self):
        '''
        when nonDiscretionary outputs or inputs presented, corresponding slack values should equal zero on specific indexes (or teteha equal to 1 in RusselModel)
        '''
        z = np.array([0])
        mep,rm = MeasureEfficiencyProportionModel(X,Y,nonDiscretionaryInputs=z),RusselModel(X,Y,nonDiscretionaryInputs=z,nonDiscretionaryOutputs=z)
        sbm,bam = SlackedBasedMeausereModel(X,Y,vrs=True,nonDiscretionaryInputs=z),BoundedAdjustedMeasureModel(X,Y,nonDiscretionaryInputs=z,nonDiscretionaryOutputs=z)
        for model in mep,rm,sbm,bam: model.solve()
        self.assertTrue(np.all(np.apply_along_axis(lambda x : x[0],1,mep.get('slack_x')) == 0))
        self.assertTrue(np.all(np.apply_along_axis(lambda x : x[0],1,sbm.get('slack_x')) == 0))
        self.assertTrue(np.all(bam.get('solution') == 1))
        self.assertTrue(np.all(np.apply_along_axis(lambda x : x[0],1,rm.get('tetha')) == 1))

        iTest = np.array([1,2])
        mep,rm = MeasureEfficiencyProportionModel(X1,Y1,nonDiscretionaryInputs=iTest),RusselModel(X1,Y1,nonDiscretionaryInputs=iTest,nonDiscretionaryOutputs=z)
        sbm,bam = SlackedBasedMeausereModel(X1,Y1,vrs=True,nonDiscretionaryInputs=iTest),BoundedAdjustedMeasureModel(X1,Y1,nonDiscretionaryInputs=iTest,nonDiscretionaryOutputs=z)
        for model in mep,rm,sbm,bam: model.solve()

        for nrInput in iTest:
            self.assertTrue(np.all(np.apply_along_axis(lambda x : x[nrInput],1,mep.get('slack_x')) == 0))
            self.assertTrue(np.all(np.apply_along_axis(lambda x : x[nrInput],1,sbm.get('slack_x')) == 0))
            self.assertTrue(np.all(np.apply_along_axis(lambda x : x[nrInput],1,rm.get('tetha')) == 1))
        for nrOutput in z:
            self.assertTrue(np.all(np.apply_along_axis(lambda x : x[nrOutput],1,bam.get('slack_y')) == 0))
            self.assertTrue(np.all(np.apply_along_axis(lambda x : x[nrOutput],1,rm.get('phi')) == 1))

if __name__ == '__main__':
    unittest.main()
