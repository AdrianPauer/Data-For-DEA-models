import cvxpy as cp
import numpy as np
import pandas as pd


class Model:
    def __init__(self, inputs, outputs, vrs = False, nonDiscretionaryInputs = None, nonDiscretionaryOutputs = None):
        self.X = inputs
        self.Y = outputs

        self.m, self.n, self.s = None, None, None
        
        self.vrs = vrs
        self.nonDisctretionaryInputs = nonDiscretionaryInputs
        self.nonDisctretionaryOutputs = nonDiscretionaryOutputs

        self.initialize_params()
        self.controlData()

        self.objective = None
        self.constraints = []
        self.variables = []

        self.problem = None
        self.solver = None
        self.solutions = dict()

    def controlData(self):
        if self.X.ndim == 1 and  isinstance(self.X[0],list) :
            raise ValueError('Input matrix is incomplete')
        if self.Y.ndim == 1 and  isinstance(self.Y[0],list) :
            raise ValueError('Output matrix is incomplete')
        if (self.X <= 0).sum() > 0 or (self.Y <= 0).sum() > 0 :
            raise ValueError('All data should be positive')
        
        if self.nonDisctretionaryInputs is not None :
            if ((self.nonDisctretionaryInputs >= self.m) | (self.nonDisctretionaryInputs < 0)).sum() > 0 : 
                raise IndexError('Index for nondiscretionary inputs should be in range [0,m-1]')
        if self.nonDisctretionaryOutputs is not None :
            if ((self.nonDisctretionaryOutputs >= self.s) | (self.nonDisctretionaryOutputs < 0)).sum() > 0 : 
                raise IndexError('Index for nondiscretionary inputs should be in range [0,s-1]')
    
    def __str__(self):
        if self.solutions is None : return 'model with no solutions'
        solutionsFrame = pd.DataFrame.from_dict(self.solutions, orient='index')
        solutionsFrame.index.name = 'DMU'
        return solutionsFrame.__str__()
    

    def scaleData(self):
        self.X = self.X/self.X.max()
        self.Y = self.Y/self.Y.max()
    

    def initialize_params(self):
        self.m = 1 if self.X.ndim == 1 else  self.X.shape[0]
        self.s = 1 if self.Y.ndim == 1 else  self.Y.shape[0]
        self.n = self.X.shape[0] if self.X.ndim == 1 else self.X.shape[1]

    def init_variables_slack_based(self):
        self.l, self.s_x, self.s_y = cp.Variable(self.n), cp.Variable(self.m), cp.Variable(self.s)
        self.variables = [self.l, self.s_x, self.s_y]
        self.solver = cp.ECOS
    
    def fill_constraints_slack_based(self, x_0 , y_0):
        self.constraints = [self.X @ self.l == x_0 - self.s_x,
                            self.Y @ self.l == y_0 + self.s_y,
                            self.s_x >= 0, self.s_y >= 0, self.l >= 0]
        if self.vrs : self.constraints.append(cp.sum(self.l) == 1)
        
        if self.nonDisctretionaryInputs is not None :
            if self.nonDisctretionaryInputs.shape[0] > 0 and self.nonDisctretionaryInputs.shape[0] <= self.m:
                self.constraints.append(self.s_x[self.nonDisctretionaryInputs] == 0)
            else : print('nonDiscInputsViolation')

        if self.nonDisctretionaryOutputs is not None :
            if self.nonDisctretionaryOutputs.shape[0] > 0 and self.nonDisctretionaryOutputs.shape[0] <= self.s:
                self.constraints.append(self.s_y[self.nonDisctretionaryOutputs] == 0)
            else : print('nonDiscOutputsViolation')
        
    def get(self, key):
        return np.array(list(map(lambda x : x[key], list(self.solutions.values()))))
        
    def solve(self):

        for i in range(self.n):
            x_0 = self.X[i] if self.m == 1 else self.X[:,i]
            y_0 = self.Y[i] if self.s == 1 else self.Y[:,i] 
            
            self.fill_constrainsts(x_0, y_0)
 
            self.problem.solve(solver = self.solver)
            self.store_results(i+1,x_0,y_0)        
        self.writeSolutionsToFile()
        return True

    
    def store_results(self,which_dmu,x_0,y_0) :
        self.solutions[which_dmu] = dict()
        for variable,key in (self.problem,'solution'),(self.s_x,'slack_x'), (self.s_y,'slack_y'), (self.l,'lambda'):
            self.solutions[which_dmu][key] = np.round(variable.value,6)
        self.solutions[which_dmu]['projectionX'] = np.round(self.X @ self.l.value,6)
        self.solutions[which_dmu]['projectionY'] = np.round(self.Y @ self.l.value,6)

        self.solutions[which_dmu]['partialEff_inputs'] =   np.round(1 - 1/x_0 * self.s_x.value,6)
        self.solutions[which_dmu]['partialEff_outputs'] =   np.round(1 + 1/y_0 * self.s_y.value,6)
    
    def writeSolutionsToFile(self):
        with open('output.txt','w') as outputFile:
            for key in self.solutions.keys():
                outputFile.write(str(key) + '\n')

                for key2,value in self.solutions[key].items():
                    
                    if isinstance(value,np.ndarray): 
                        outputFile.write(' '.join(list(map(str,value))) + '\n')
                    else : outputFile.write(str(value) + '\n')
        outputFile.close()



    


class AdditiveModel(Model):
    def __init__(self, inputs, outputs, vrs = False, nonDiscretionaryInputs = None, nonDiscretionaryOutputs = None):
        Model.__init__(self,inputs, outputs,vrs, nonDiscretionaryInputs, nonDiscretionaryOutputs)
        Model.init_variables_slack_based(self)

        self.objective = lambda s_x, s_y : 1 - (cp.sum(s_x) + cp.sum(s_y))

        print('Result might be inacurete due to working with nonScaledData')
    
    def fill_constrainsts(self,x_0,y_0):
        Model.fill_constraints_slack_based(self, x_0, y_0)
        self.problem = cp.Problem(cp.Minimize(self.objective(self.s_x, self.s_y)),self.constraints)
        

    


class WeightedAdditiveModel(Model):
    def __init__(self, inputs, outputs, inputWeights, outputWeights, vrs = False, nonDiscretionaryInputs = None, nonDiscretionaryOutputs = None):
        Model.__init__(self,inputs, outputs, vrs, nonDiscretionaryInputs, nonDiscretionaryOutputs)
        Model.init_variables_slack_based(self)

        self.w_x,self.w_y = inputWeights, outputWeights
        self.objective = lambda s_x, s_y, w_x, w_y :  1 - (w_x @ s_x + w_y @ s_y)

        print('Result might be inacurete due to working with nonScaledData')
    
    def fill_constrainsts(self,x_0,y_0):
        Model.fill_constraints_slack_based(self, x_0, y_0)
        self.problem = cp.Problem(cp.Minimize(self.objective(self.s_x, self.s_y,self.w_x, self.w_y)),self.constraints)
    

class MeasureEfficiencyProportionModel(Model):
    def __init__(self, inputs, outputs, vrs = False, nonDiscretionaryInputs = None, nonDiscretionaryOutputs = None):
        Model.__init__(self,inputs, outputs,vrs, nonDiscretionaryInputs, nonDiscretionaryOutputs)
        Model.init_variables_slack_based(self)
        self.scaleData()
        self.objective = lambda s_x, s_y, x_0, y_0: 1 - (1/(self.m + self.s)) * (cp.sum(s_x/x_0) + cp.sum(s_y/y_0) )
    
    def fill_constrainsts(self,x_0,y_0):
        Model.fill_constraints_slack_based(self, x_0, y_0)
        self.problem = cp.Problem(cp.Minimize(self.objective(self.s_x, self.s_y,x_0,y_0)),self.constraints)
    
class BoundedAdjustedMeasureModel(Model):
    def __init__(self, inputs, outputs, vrs = False, nonDiscretionaryInputs = None, nonDiscretionaryOutputs = None):
        Model.__init__(self,inputs, outputs,vrs, nonDiscretionaryInputs, nonDiscretionaryOutputs)
        Model.init_variables_slack_based(self)
        self.scaleData()

        self.objective = lambda s_x,s_y,x_0,y_0 : self.objective_function(s_x,s_y,x_0,y_0)
    
    def objective_function(self,s_x,s_y,x_0,y_0):
        x_ = self.X.min() if self.m == 1 else self.X.min(axis = 1)
        y_ = self.Y.max() if self.s == 1 else self.Y.min(axis = 1)

        sum_x, sum_y = None, None

        diffrence_x = x_0 - x_
        non_zero_x = x_0[diffrence_x != 0]
        if non_zero_x.size == 0 : sum_x = 0
        else : sum_x = cp.sum(s_x[np.where(diffrence_x != 0)]/(non_zero_x))

        diffrence_y = y_0 - y_
        non_zero_y = y_0[diffrence_y != 0]
        if non_zero_y.size == 0 : sum_y = 0
        else : sum_y = cp.sum(s_y[np.where(diffrence_y != 0)]/(non_zero_y))

        return 1 - 1/(self.m + self.s) * (sum_x + sum_y)
    
    def fill_constrainsts(self,x_0,y_0):
        Model.fill_constraints_slack_based(self, x_0, y_0)
        self.problem = cp.Problem(cp.Minimize(self.objective(self.s_x, self.s_y,x_0,y_0)),self.constraints)
    

class RusselModel(Model):
    def __init__(self, inputs, outputs, vrs = False, nonDiscretionaryInputs = None, nonDiscretionaryOutputs = None):
        Model.__init__(self,inputs, outputs,vrs, nonDiscretionaryInputs, nonDiscretionaryOutputs)
        self.scaleData()
        self.l =  cp.Variable(self.n)
        self.tetha, self.psi, self.phi = cp.Variable(self.m),cp.Variable(self.s),cp.Variable(self.s)

        self.objective = lambda tetha,psi : 1/(self.m+self.s) * (cp.sum(tetha) + cp.sum(psi))
        self.solver = cp.SCS
    
    
    def fill_constrainsts(self, x_0, y_0):
        self.constraints = [np.ones(self.m) - self.tetha >= np.zeros(self.m),
                            self.phi - np.ones(self.s) >= np.zeros(self.s),
                            self.l >= np.zeros(self.n)]
        
        if self.vrs : self.constraints.append(cp.sum(self.l) == 1)
        
        if self.m == 1 :
            self.constraints.append(-self.X@self.l + x_0 * self.tetha >= np.zeros(self.m))
        else : self.constraints.append(-self.X@self.l + np.diag(x_0)@self.tetha >= np.zeros(self.m))

        
        if self.s == 1 :
            self.constraints.append(self.Y@self.l  - y_0 * self.phi >= np.zeros(self.s))
        else : self.constraints.append(self.Y@self.l  - np.diag(y_0)@self.phi >= np.zeros(self.s))

        
        self.constraints.append(cp.bmat( [[cp.diag(self.psi),np.eye(self.s)], [np.eye(self.s),cp.diag(self.phi)] ]) >> 0)
            
        if self.nonDisctretionaryInputs is not None :
            if self.nonDisctretionaryInputs.shape[0] > 0 and self.nonDisctretionaryInputs.shape[0] <= self.m:
                self.constraints.append(self.tetha[self.nonDisctretionaryInputs] == 1)
            else : print('nonDiscInputsViolation')

        if self.nonDisctretionaryOutputs is not None :
            if self.nonDisctretionaryOutputs.shape[0] > 0 and self.nonDisctretionaryOutputs.shape[0] <= self.s:
                self.constraints.append(self.phi[self.nonDisctretionaryOutputs] == 1)
            else : print('nonDiscOutputsViolation')
        
            
        self.problem = cp.Problem(cp.Minimize(self.objective(self.tetha, self.psi)),self.constraints)
        
    
    def store_results(self, which_dmu,x_0, y_0):
        self.solutions[which_dmu] = dict()
        for variable,key in (self.problem,'solution'),(self.tetha,'tetha'), (self.phi,'phi'), (self.psi,'psi'), (self.l,'lambda'):
            self.solutions[which_dmu][key] = np.round(variable.value,4)

        self.solutions[which_dmu]['projectionX'] = np.round(self.X @ self.l.value,6)
        self.solutions[which_dmu]['projectionY'] = np.round(self.Y @ self.l.value,6)
        
        self.solutions[which_dmu]['partialEff_inputs'] =  np.round(self.tetha.value,4)
        self.solutions[which_dmu]['partialEff_outputs'] =  np.round(1/self.phi.value,4)
    
    
class SlackedBasedMeausereModel(Model):
    def __init__(self, inputs, outputs, vrs = False, nonDiscretionaryInputs = None, nonDiscretionaryOutputs = None):
        Model.__init__(self,inputs, outputs,vrs, nonDiscretionaryInputs, nonDiscretionaryOutputs)
        Model.init_variables_slack_based(self)
        self.scaleData()
        
        self.t = cp.Variable(1)
        self.objective = lambda t,x_0 : t - (1/self.m *cp.sum(self.s_x/x_0)) 
    
    
    def fill_constrainsts(self, x_0, y_0):
        self.constraints = [ self.t + 1/self.s * cp.sum(self.s_y/y_0) == 1,
                            self.X@self.l + self.s_x == self.t * x_0,
                            self.Y@self.l - self.s_y == self.t * y_0,
                            self.s_x >= 0, self.s_y >= 0, self.l >= 0]
        
        if self.vrs : self.constraints.append(cp.sum(self.l) == self.t)

        if self.nonDisctretionaryInputs is not None :
            if self.nonDisctretionaryInputs.shape[0] > 0 and self.nonDisctretionaryInputs.shape[0] <= self.m:
                self.constraints.append(self.s_x[self.nonDisctretionaryInputs] == 0)
            else : print('nonDiscInputsViolation')

        if self.nonDisctretionaryOutputs is not None :
            if self.nonDisctretionaryOutputs.shape[0] > 0 and self.nonDisctretionaryOutputs.shape[0] <= self.s:
                self.constraints.append(self.s_y[self.nonDisctretionaryOutputs] == 0)
            else : print('nonDiscOutputsViolation')
        
        
        self.problem = cp.Problem(cp.Minimize(self.objective(self.t, x_0)),self.constraints)

    
    def store_results(self, which_dmu,x_0,y_0):
        self.solutions[which_dmu] = dict()
        for variable,key in (self.problem,'solution'),(self.s_x,'slack_x'), (self.s_y,'slack_y'), (self.t,'t'),(self.l,'lambda'):
            self.solutions[which_dmu][key] = np.round(variable.value,6)
        self.solutions[which_dmu]['projectionX'] = np.round(self.X @ self.l.value,6)
        self.solutions[which_dmu]['projectionY'] = np.round(self.Y @ self.l.value,6)

        self.solutions[which_dmu]['partialEff_inputs'] =   np.round(1 - 1/x_0 * self.s_x.value,6)
        self.solutions[which_dmu]['partialEff_outputs'] =   np.round(1 + 1/y_0 * self.s_y.value,6)
    
