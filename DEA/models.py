import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class Model:
    def __init__(self, input_file_path, output_file_path, vrs = False, nonDiscretionaryInputs = None, nonDiscretionaryOutputs = None):
        self.X, self.Y = self.getDataFromFile(input_file=input_file_path,output_file=output_file_path)
        self.scale_factor = 1

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
        self.columns_to_write = ['projectionX','projectionY', 'slack_x', 'slack_y','lambda','ratio_s_x','ratio_s_y']

    def controlData(self):
        if self.X.ndim == 1 and  isinstance(self.X[0],list) :
            raise ValueError('Input matrix is incomplete <-> input all element of m x n matrix')
        if self.Y.ndim == 1 and  isinstance(self.Y[0],list) :
            raise ValueError('Output matrix is incomplete <-> input all element of m x n matrix')
        if (self.X <= 0).sum() > 0 or (self.Y <= 0).sum() > 0 :
            raise ValueError('All data should be positive')
        
        # n is set according to X.shape[0]
        Y_shape = self.X.shape[0] if self.Y.ndim == 1 else self.Y.shape[1]
        if self.n != Y_shape:
            raise ValueError('input matrix and output matrix should have same second dimension')


        if self.nonDisctretionaryInputs is not None :
            if ((self.nonDisctretionaryInputs >= self.m) | (self.nonDisctretionaryInputs < 0)).sum() > 0 : 
                raise IndexError('Index for nondiscretionary inputs should be in range [0,m-1]')
        if self.nonDisctretionaryOutputs is not None :
            if ((self.nonDisctretionaryOutputs >= self.s) | (self.nonDisctretionaryOutputs < 0)).sum() > 0 : 
                raise IndexError('Index for nondiscretionary inputs should be in range [0,s-1]')
    
    def __str__(self):
        if self.solutions is None : return 'model with no solutions'
        return self.solutionsFrame.__str__()
    

    def scaleData(self):
        self.scale_factor = max(self.X.max(), self.Y.max())
        self.X = self.X/self.scale_factor
        self.Y = self.Y/self.scale_factor
    

    def initialize_params(self):
        # self.m = 1 if self.X.ndim == 1 else  self.X.shape[0]
        # self.s = 1 if self.Y.ndim == 1 else  self.Y.shape[0]
        self.m = self.X.shape[0]
        self.s = self.Y.shape[0]
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
            # x_0 = self.X[i] if self.m == 1 else self.X[:,i]
            # y_0 = self.Y[i] if self.s == 1 else self.Y[:,i] 

            x_0 = self.X[:,i]
            y_0 = self.Y[:,i] 
            
            self.fill_constrainsts(x_0, y_0)
 
            self.problem.solve(solver = self.solver)
            self.store_results(i+1,x_0,y_0)
        # compute rank and reference sets after solving LP
        for i in range(1, self.n +1):
            self.solutions[i]['in_reference_set'] = np.where(self.get('lambda')[:,i-1] > 0)[0] + 1
        
        self.solutionsFrame = pd.DataFrame.from_dict(self.solutions, orient='index')
        self.solutionsFrame.index.name = 'DMU'
        self.solutionsFrame['rank'] = self.solutionsFrame['solution'].rank(method = 'min', ascending=False)

        for key in self.solutions.keys(): 
            self.solutions[key]['rank'] = self.solutionsFrame['rank'][key] 
        return True

    
    def store_results(self,which_dmu,x_0,y_0) :
        self.solutions[which_dmu] = dict()
        for variable,key in (self.problem,'solution'),(self.s_x,'slack_x'), (self.s_y,'slack_y'), (self.l,'lambda'):
            self.solutions[which_dmu][key] = np.round(variable.value,6)
        self.solutions[which_dmu]['projectionX'] = np.round(self.X @ self.l.value,6) * self.scale_factor
        self.solutions[which_dmu]['projectionY'] = np.round(self.Y @ self.l.value,6) * self.scale_factor

        self.solutions[which_dmu]['partialEff_inputs'] =   np.round(1 - 1/x_0 * self.s_x.value,6)
        self.solutions[which_dmu]['partialEff_outputs'] =   np.round(1 + 1/y_0 * self.s_y.value,6)
        
        distance_to_projection_x = np.sqrt(((x_0 - self.solutions[which_dmu]['projectionX'])**2).sum())
        overal_distance_x = np.sqrt(((self.solutions[which_dmu]['projectionX'])**2).sum())
        self.solutions[which_dmu]['ratio_s_x'] = distance_to_projection_x / overal_distance_x

        distance_to_projection_y = np.sqrt(((y_0 - self.solutions[which_dmu]['projectionY'])**2).sum())
        overal_distance_y = np.sqrt(((self.solutions[which_dmu]['projectionY'])**2).sum())
        self.solutions[which_dmu]['ratio_s_y'] = distance_to_projection_y / overal_distance_y
        
    
    def writeSolutionsToFile(self, output_file):
        if self.solutions is None : return 'model with no solutions yet'
        with open(output_file,'w') as outputFile:
            # a = self.solutionsFrame.to_csv(None, columns = ['projectionX','projectionY', 'slack_x', 'slack_y','lambda','ratio_s_x','ratio_s_y'],lineterminator='\n')
            # print(a.__repr__())
            outputFile.write(self.solutionsFrame.to_string(columns = self.columns_to_write))
        outputFile.close()
    
    def getDataFromFile(self,input_file,output_file):
        try:
            with open(input_file,'r') as input:
                X = np.array(input.readlines())
                X = np.array(list(map(lambda x : list(map(lambda number : float(number),x.split())),X)))

            with open(output_file,'r') as output:
                Y = np.array(output.readlines())
                Y = np.array(list(map(lambda x : list(map(lambda number : float(number),x.split())),Y)))
            return X,Y
        except ValueError:
                raise ValueError('All data should be convertable to float')
    
    def plot_projections(self,display = True, file = None):
        if self.m != 1 or self.s != 1 : 
            raise('DMUs can not be visualised due to high dimension')
        
        fig = Figure()
        ax = fig.subplots()
        # plot effective frontier
        inputs = self.X[0,:] * self.scale_factor
        outputs = self.Y[0,:] * self.scale_factor

        projX = self.get('projectionX')[:,0] 
        projY = self.get('projectionY')[:,0] 

        
        effective_indexes = np.arange(self.n)[self.get('solution') == 1]
        effective_inputs = inputs[effective_indexes]
        effective_outputs = outputs[effective_indexes]

        if self.vrs == True:
            sorted_indexes = np.argsort(effective_inputs)
            sorted_inputs = effective_inputs[sorted_indexes]
            sorted_outputs = effective_outputs[sorted_indexes]

            ax.plot(sorted_inputs, sorted_outputs, label = 'vrs')
            ax.plot([sorted_inputs[0],sorted_inputs[0]],[0,sorted_outputs[0]],'b', linestyle = '--',linewidth = 1)
            ax.plot([sorted_inputs[-1],sorted_inputs[-1] + 5],[sorted_outputs[-1],sorted_outputs[-1]],'b', linestyle = '--',linewidth = 1)
        
        else:
            slope = effective_outputs[0]/effective_inputs[0]
            mx = inputs.max() + 2
            ax.plot(np.arange(mx),slope * np.arange(mx),color = 'blue',linewidth = 1,alpha =0.8, label = 'crs')

        # plot units and projections
        for i in range(self.n):
            ax.plot([inputs[i],projX[i]],[outputs[i],projY[i]],'.--',color = 'black',markersize=10)
            if i not in effective_indexes:
                ax.plot(projX[i],projY[i],'.',color= 'red')
            txt = chr(ord('A') + i)
            ax.text(inputs[i] + 0.2, outputs[i],txt)
        ax.set_xlabel('vstupy')
        ax.set_ylabel('výstupy')
        ax.set_title(self.__class__.__name__ )
        ax.grid(alpha = 0.1)

        ax.legend()
        fig.savefig(file)
    
    def plot_effectivity(self, file):
        fig = Figure()
        ax = fig.subplots()

        labels = list(map(lambda x : chr(ord('A') + x - 1), self.solutions.keys()))
        bar1 = ax.bar(labels, self.get('solution'), width = 0.4)
        for rect in bar1 :
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')

        ax.set_xlabel("DMU")
        ax.set_ylabel("Efektivita")
        ax.set_title("Efektivita")

        fig.savefig(file)

    def plot_DMU_attributes(self, which, file_path): 
        # partial efficiencies
        fig = Figure()
        ax = fig.subplots(1,2,sharey=True)
        part_eff_x = self.solutions[which]['partialEff_inputs']
        part_eff_y = self.solutions[which]['partialEff_outputs']
        bar1 = ax[0].bar(np.arange(1, self.m+1), part_eff_x, width = 0.4, color= 'blue')
        
        bar2 = ax[1].bar(np.arange(1, self.s+1), part_eff_y, width = 0.4, color= 'blue')

        for rect in bar1 :
            height = rect.get_height()
            ax[0].text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')
        for rect in bar2 :
            height = rect.get_height()
            ax[1].text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')
            
        
        ax[0].set_ylabel("čiastková efektivita")
        fig.suptitle("čiastková efektivita vstupov a výstupov")
        ax[0].set_xlabel("vstup")
        ax[1].set_xlabel("výstup")
        fig.savefig(file_path + 'paratial_efficiencies_' + str(which) + '.png')

        #reference set
        fig = Figure()
        ax = fig.subplots()
        l = self.solutions[which]['lambda']
        ref_set = np.where(l > 0)[0] + 1
        labels = list(map(lambda x:'DMU' + str(x), ref_set))

        ax.pie(l[ref_set-1], labels=labels, autopct='%1.1f%%', shadow = True)
        ax.set_title('Referenčná množina ')
        fig.savefig(file_path + 'reference_set_' + str(which) + '.png' )

        #%s_x a %s_y

class AdditiveModel(Model):
    def __init__(self, input_file_path, output_file_path, vrs = False, nonDiscretionaryInputs = None, nonDiscretionaryOutputs = None):
        Model.__init__(self, input_file_path, output_file_path, vrs, nonDiscretionaryInputs, nonDiscretionaryOutputs)
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

class RangeAdjustedMeasureModel(WeightedAdditiveModel):
    def __init__(self,inputs, outputs, vrs = False, nonDiscretionaryInputs = None, nonDiscretionaryOutputs = None):
        WeightedAdditiveModel.__init__(self,inputs, outputs, None, None, vrs, nonDiscretionaryInputs, nonDiscretionaryOutputs)
        x_min = np.array([self.X.min()]) if self.m == 1 else self.X.min(axis = 1)
        x_max = np.array([self.X.max()]) if self.m == 1 else self.X.max(axis = 1)

        y_min = np.array([self.Y.min()]) if self.s == 1 else self.Y.min(axis = 1)
        y_max = np.array([self.Y.max()]) if self.s == 1 else self.Y.max(axis = 1)

        self.w_x = 1 / ( (x_max - x_min) * (self.m + self.s) )
        self.w_y = 1 / ( (y_max - y_min) * (self.m + self.s) )

        

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

        self.columns_to_write = ['projectionX','projectionY', 'tetha', 'phi','psi','lambda','ratio_s_x','ratio_s_y']
    
    
    def fill_constrainsts(self, x_0, y_0):
        self.constraints = [np.ones(self.m) - self.tetha >= np.zeros(self.m),
                            self.phi - np.ones(self.s) >= np.zeros(self.s),
                            self.l >= np.zeros(self.n)]
        
        if self.vrs : self.constraints.append(cp.sum(self.l) == 1)
        
        # if self.m == 1 :
        #     self.constraints.append(-self.X@self.l + x_0 * self.tetha >= np.zeros(self.m))
        # else : self.constraints.append(-self.X@self.l + np.diag(x_0)@self.tetha >= np.zeros(self.m))

        self.constraints.append(-self.X@self.l + np.diag(x_0)@self.tetha >= np.zeros(self.m))


        
        # if self.s == 1 :
        #     self.constraints.append(self.Y@self.l  - y_0 * self.phi >= np.zeros(self.s))
        # else : self.constraints.append(self.Y@self.l  - np.diag(y_0)@self.phi >= np.zeros(self.s))

        self.constraints.append(self.Y@self.l  - np.diag(y_0)@self.phi >= np.zeros(self.s))

        
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

        self.solutions[which_dmu]['projectionX'] = np.round(self.X @ self.l.value,6) * self.scale_factor
        self.solutions[which_dmu]['projectionY'] = np.round(self.Y @ self.l.value,6) * self.scale_factor
        
        self.solutions[which_dmu]['partialEff_inputs'] =  np.round(self.tetha.value,4)
        self.solutions[which_dmu]['partialEff_outputs'] =  np.round(1/self.phi.value,4)
        
        distance_to_projection_x = np.sqrt(((x_0 - self.solutions[which_dmu]['projectionX'])**2).sum())
        overal_distance_x = np.sqrt(((x_0)**2).sum())
        self.solutions[which_dmu]['ratio_s_x'] = distance_to_projection_x / overal_distance_x

        distance_to_projection_y = np.sqrt(((y_0 - self.solutions[which_dmu]['projectionY'])**2).sum())
        overal_distance_y = np.sqrt(((y_0)**2).sum())
        self.solutions[which_dmu]['ratio_s_y'] = distance_to_projection_y / overal_distance_y
    
    
class SlackedBasedMeausereModel(Model):
    def __init__(self, inputs, outputs, vrs = False, nonDiscretionaryInputs = None, nonDiscretionaryOutputs = None):
        Model.__init__(self,inputs, outputs,vrs, nonDiscretionaryInputs, nonDiscretionaryOutputs)
        Model.init_variables_slack_based(self)
        self.scaleData()
        
        self.t = cp.Variable(1)
        self.objective = lambda t,x_0 : t - (1/self.m *cp.sum(self.s_x/x_0)) 
    
    
    def fill_constrainsts(self, x_0, y_0):
        if self.s == 1 : x_0, y_0 = x_0[0],y_0[0]
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
        self.solutions[which_dmu]['solution'] = np.round(self.problem.value, 6)
        for variable,key in (self.s_x,'slack_x'), (self.s_y,'slack_y'),(self.l,'lambda'):
            self.solutions[which_dmu][key] = np.round(variable.value / self.t.value, 6)
        self.solutions[which_dmu]['t'] = self.t.value
        
        self.solutions[which_dmu]['projectionX'] = np.round(self.X @ self.l.value / self.t.value, 6) * self.scale_factor 
        self.solutions[which_dmu]['projectionY'] = np.round(self.Y @ self.l.value / self.t.value, 6) * self.scale_factor

        self.solutions[which_dmu]['partialEff_inputs'] =   np.round(1 - 1/x_0 * self.s_x.value,6)
        self.solutions[which_dmu]['partialEff_outputs'] =   np.round(1 + 1/y_0 * self.s_y.value,6)
        
        distance_to_projection_x = np.sqrt(((x_0 - self.solutions[which_dmu]['projectionX'])**2).sum())
        overal_distance_x = np.sqrt(((x_0)**2).sum())
        self.solutions[which_dmu]['ratio_s_x'] = distance_to_projection_x / overal_distance_x

        distance_to_projection_y = np.sqrt(((y_0 - self.solutions[which_dmu]['projectionY'])**2).sum())
        overal_distance_y = np.sqrt(((y_0)**2).sum())
        self.solutions[which_dmu]['ratio_s_y'] = distance_to_projection_y / overal_distance_y

class GRAM(Model):
     def __init__(self, inputs, outputs, p = 1, vrs = False, nonDiscretionaryInputs = None, nonDiscretionaryOutputs = None):
        Model.__init__(self,inputs, outputs,vrs, nonDiscretionaryInputs, nonDiscretionaryOutputs)
        Model.init_variables_slack_based(self)
        self.scaleData()

        self.p = p
        x_min = self.X.min() if self.m == 1 else self.X.min(axis = 1)
        x_max = self.X.max() if self.m == 1 else self.X.max(axis = 1)
        x_dif = x_max - x_min

        non_zero_dif = x_dif[x_dif != 0]
        if non_zero_dif.size == 0 : sum_x = 0
        else : sum_x = cp.sum((self.s_x[np.where(x_dif != 0)]/(non_zero_dif))**self.p)

        y_min = self.Y.min() if self.s == 1 else self.Y.min(axis = 1)
        y_max = self.Y.max() if self.s == 1 else self.Y.max(axis = 1)
        y_dif = y_max - y_min

        non_zero_dif = y_dif[y_dif != 0]
        if non_zero_dif.size == 0 : sum_y = 0
        else : sum_y = cp.sum((self.s_y[np.where(y_dif != 0)]/(non_zero_dif))**self.p)

        self.objective = lambda:  1 - ((1/(self.m + self.s))**(1/self.p)) * ((sum_y + sum_x)**(1/self.p))

     def fill_constrainsts(self,x_0,y_0):
        Model.fill_constraints_slack_based(self, x_0, y_0)
        self.problem = cp.Problem(cp.Minimize(self.objective()),self.constraints)


