import cvxpy as cp
import numpy as np
import pandas as pd

class Model:
    def __init__(self, inputs, outputs):
        self.X = inputs
        self.Y = outputs
        self.m, self.n, self.s = None, None, None
        self.initialize_params()

        self.objective = None
        self.constraints = []
        self.variables = []

        self.problem = None
        self.solver = None
        self.solutions = dict()

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
                            cp.sum(self.l) == 1,
                            self.s_x >= 0, self.s_y >= 0, self.l >= 0]
        
    def solve(self):

        for i in range(self.n):
            x_0 = self.X[i] if self.m == 1 else self.X[:,i]
            y_0 = self.Y[i] if self.s == 1 else self.Y[:,i] 
            
            self.fill_constrainsts(x_0, y_0)
 
            self.problem.solve(solver = self.solver)
            self.store_results(i+1)

        solutions_Frame = pd.DataFrame.from_dict(self.solutions, orient='index')
        solutions_Frame.index.name = 'DMU'
        print(solutions_Frame)
    
    def store_results(self,which_dmu) :
        self.solutions[which_dmu] = dict()
        for variable,key in (self.problem,'solution'),(self.s_x,'slack_x'), (self.s_y,'slack_y'), (self.l,'lambda'):
            self.solutions[which_dmu][key] = variable.value

class AdditiveModel(Model):
    def __init__(self, inputs, outputs):
        Model.__init__(self,inputs, outputs)
        Model.init_variables_slack_based(self)

        self.objective = lambda s_x, s_y : 1 - (cp.sum(s_x) + cp.sum(s_y))
    
    def fill_constrainsts(self,x_0,y_0):
        Model.fill_constraints_slack_based(self, x_0, y_0)
        self.problem = cp.Problem(cp.Minimize(self.objective(self.s_x, self.s_y)),self.constraints)
    
    def store_results(self,which_dmu) :
        self.solutions[which_dmu] = dict()
        for variable,key in (self.problem,'solution'),(self.s_x,'slack_x'), (self.s_y,'slack_y'), (self.l,'lambda'):
            self.solutions[which_dmu][key] = variable.value


    


class WeightedAdditiveModel(Model):
    def __init__(self, inputs, outputs, inputWeights, outputWeights):
        Model.__init__(self,inputs, outputs)
        Model.init_variables_slack_based(self)

        self.w_x,self.w_y = inputWeights, outputWeights
        self.objective = lambda s_x, s_y, w_x, w_y :  1 - (w_x @ s_x + w_y @ s_y)
    
    def fill_constrainsts(self,x_0,y_0):
        Model.fill_constraints_slack_based(self, x_0, y_0)
        self.problem = cp.Problem(cp.Minimize(self.objective(self.s_x, self.s_y,self.w_x, self.w_y)),self.constraints)
    

class MeasureEfficiencyProportionModel(Model):
    def __init__(self, inputs, outputs):
        Model.__init__(self,inputs, outputs)
        Model.init_variables_slack_based(self)
        self.objective = lambda s_x, s_y, x_0, y_0: 1 - (1/(self.m + self.s)) * (cp.sum(s_x/x_0) + cp.sum(s_y/y_0) )
    
    def fill_constrainsts(self,x_0,y_0):
        Model.fill_constraints_slack_based(self, x_0, y_0)
        self.problem = cp.Problem(cp.Minimize(self.objective(self.s_x, self.s_y,x_0,y_0)),self.constraints)
    
class BoundedAdjustedMeasureModel(Model):
    def __init__(self, inputs, outputs):
        Model.__init__(self,inputs, outputs)
        Model.init_variables_slack_based(self)

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
    def __init__(self, inputs, outputs):
        Model.__init__(self,inputs, outputs)
        self.l =  cp.Variable(self.n)
        self.tetha, self.psi, self.phi = cp.Variable(self.m),cp.Variable(self.s),cp.Variable(self.s)

        self.objective = lambda tetha,psi : 1/(self.m+self.s) * (cp.sum(tetha) + cp.sum(psi))
        self.solver = cp.SCS
    
    
    def fill_constrainsts(self, x_0, y_0):
        self.constraints = [np.ones(self.m) - self.tetha >= np.zeros(self.m),
                            self.phi - np.ones(self.s) >= np.zeros(self.s),
                            self.l >= np.zeros(self.n)]
        
        if self.m == 1 :
            self.constraints.append(-self.X@self.l + x_0 * self.tetha >= np.zeros(self.m))
        else : self.constraints.append(-self.X@self.l + np.diag(x_0)@self.tetha >= np.zeros(self.m))

        
        if self.s == 1 :
            self.constraints.append(self.Y@self.l  - y_0 * self.phi >= np.zeros(self.s))
        else : self.constraints.append(self.Y@self.l  - np.diag(y_0)@self.phi >= np.zeros(self.s))

        
        for r in range(self.s):
            self.constraints.append(cp.bmat([ [self.psi[r], 1],
                                              [1, self.phi[r]] ]) >> 0)
            
        self.problem = cp.Problem(cp.Minimize(self.objective(self.tetha, self.psi)),self.constraints)
    
    def store_results(self, which_dmu):
        self.solutions[which_dmu] = dict()
        for variable,key in (self.problem,'solution'),(self.tetha,'tetha'), (self.phi,'phi'), (self.psi,'psi'), (self.l,'lambda'):
            self.solutions[which_dmu][key] = variable.value

    
    

    
    

class SlackedBasedMeausereModel(Model):
    def __init__(self, inputs, outputs):
        Model.__init__(self,inputs, outputs)
        Model.init_variables_slack_based(self)
        
        self.t = cp.Variable(1)
        self.objective = lambda t,x_0 : t - (1/self.m *cp.sum(self.s_x/x_0)) 
    
    
    def fill_constrainsts(self, x_0, y_0):
        self.constraints = [ self.t + 1/self.s * cp.sum(self.s_y/y_0) == 1,
                            self.X@self.l + self.s_x == self.t * x_0,
                            self.Y@self.l - self.s_y == self.t * y_0,
                            cp.sum(self.l) == self.t,
                            self.s_x >= 0, self.s_y >= 0, self.l >= 0]
        self.problem = cp.Problem(cp.Minimize(self.objective(self.t, x_0)),self.constraints)
    
    def store_results(self, which_dmu):
        self.solutions[which_dmu] = dict()
        for variable,key in (self.problem,'solution'),(self.s_x,'slack_x'), (self.s_y,'slack_y'), (self.t,'t'),(self.l,'lambda'):
            self.solutions[which_dmu][key] = variable.value
    
