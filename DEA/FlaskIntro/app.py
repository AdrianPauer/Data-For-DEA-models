from flask import Flask, render_template, url_for, request, redirect, flash, send_from_directory
from werkzeug.utils import secure_filename
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from models import *

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
app.secret_key = "super secret key"

model_properties = {'p1': 'Projekcia (x&#770; = x<sub>o</sub> - s<sub>x</sub>  , y&#770; = y<sub>o</sub> - s<sub>y</sub>) útvaru leží na efektívnej hranici množiny produkčnýych možností',
                    'p2': 'Ak je hodnota účelovej rovná 1, potom je útvar efektívny',
                    'p3': 'Super efetivita:',
                    'p4': '''Striktná monotónnosť efektivity: Ak útvar D dominuje útvaru C, tak efektivita útvaru D je väčšia ako hodnota efektivity útvaru C.
                             Táto podmienka je splnená ak účelová funkcia úlohy nezávisí od (x<sub>o</sub>,y<sub>o</sub>).''',
                    'p5': 'Optimálne riešenie úolhy, a teda aj efektivita je ohraničená zdola nulou.',
                    'p6': 'Invariantnosť vzhľadom na zmenu jednotiek: Optimálne riešenie úlohy sa nezmení, ak niektorý vstup alebo výstup všetkých útvarov prenásobíme kladnou konštantou.',
                    'p7': 'Invariantnosť vzhľadom na posun: Optimálne riešenie úlohy sa nezmení, ak ku niektorému vstupu alebo výstupu všetkých útvarov pripočítame kladnú konštantu.',
                    'p8': 'Unikátnosť projekcie útvaru. Podmienka platí, ak je účelová funkcia rýdzo konvexná.'}
# parameters of model
available_models = {'AD' : { 'model': AdditiveModel,
                            'name': 'additívny',
                             'popis': [model_properties['p1'], model_properties['p2' ], model_properties['p3'],model_properties['p4'], model_properties['p7'],
                                      'Model nie je invariantný vzhľadom na zmenu jednotiek.', 'Optimálne riešenie úlohy môže byť záporné',
                                    '(x<sub>o</sub>,y<sub>o</sub>) môže mať viacero projekcií.']},

                    'WAD': { 'model': WeightedAdditiveModel,
                            'name' : 'vážený additívny',
                            'popis': [model_properties['p1'], model_properties['p2' ], model_properties['p3'],model_properties['p4'], model_properties['p7'],
                                      'Pre vhodnú voľbu váh sa model stane invariantný vzhľadom na zmenu jednotiek, pričom invariantnosť vzhľadom na posun sa zachová.']},
                    'RAM':{ 'model': RangeAdjustedMeasureModel,
                            'name' : 'RAM - Range Ajusted Measure Model',
                            'popis': [model_properties['p1'], model_properties['p2' ], model_properties['p3'],model_properties['p4'], model_properties['p7'],
                                      model_properties['p6']]},

                    'MEP': {'model': MeasureEfficiencyProportionModel,
                            'name' : 'MEP - Measure Efficiency Proportion ',
                            'popis': ['Model možno vnímať ako WAD model so špecifickými hodnotami váh. Váhy závisia od vstupov a výstupov (x<sub>o</sub>,y<sub>o</sub>), teda sú špecifické pre každý útvar.',
                                      'Váhy sú definované pre kladné dáta.', model_properties['p1'], model_properties['p2' ], model_properties['p3'],model_properties['p4'],
                                      model_properties['p6']]},

                    'BAM': {'model': BoundedAdjustedMeasureModel,
                            'name' : 'BAM - Bounded Adjusted Measure',
                            'popis' : ['Model patrí do triedy WAD so špecifickými hodnotami váh.',
                                    'Pokiaľ je potrebné v účelovej funkcii deliť nulou, model redukuje dimenziu problému a problémové vstupy a výstupy ignoruje.',
                                     model_properties['p1'], model_properties['p2' ], model_properties['p3'],model_properties['p4'],model_properties['p6'], model_properties['p7'],                                    
                                     ]},

                    'SBM': {'model': SlackedBasedMeausereModel,
                            'name' : 'SBM - Slacked Based Meausere',
                            'popis': [model_properties['p1'], model_properties['p2' ], model_properties['p3'],model_properties['p4'],model_properties['p5'], model_properties['p6' ], 
                                      'Model nie je invariantný vzhľadom na posun.']},

                    'RM': {'model': RusselModel,
                           'name': 'Ruselov model',
                           'popis': [model_properties['p1'], model_properties['p2' ], model_properties['p3'],model_properties['p4'],
                                     model_properties['p5'], model_properties['p6' ], 'Model nie je invariantný vzhľadom na posun.']},
                    'GRAM': {'model': GRAM,
                           'name': 'GRAM model',
                           'popis': [model_properties['p1'], model_properties['p2' ], model_properties['p3'],model_properties['p5']]}
                    }


class Parameters():
    def __init__(self) :
        self.vrs = False
        self.chosen_model = None
        self.graph = False
        self.check_inputs, self.check_outputs = None, None
        self.inputs_file_path, self.outputs_file_path = None, None
        self.solution = None
        self.images= None
        
        # WAD and GRAM models need to be treated somehow diffrent
        self.wad = False
        self.weights_inputs, self.weights_outputs = None, None
        self.p = 1

    def __str__(self):
        return (f"Parameters:\n"
                f"vrs: {self.vrs}\n"
                f"chosen_model: {self.chosen_model}\n"
                f"graph: {self.graph}\n"
                f"check_inputs: {self.check_inputs}\n"
                f"check_outputs: {self.check_outputs}\n"
                f"inputs_file_path: {self.inputs_file_path}\n"
                f"outputs_file_path: {self.outputs_file_path}")

params = Parameters()

@app.route('/', methods = ['POST', 'GET'])
def index():
    if request.method ==  'POST':
        if request.form['vynosy'] == 'vrs' : params.vrs = True
        else: params.vrs = False

        if request.form['graph'] == 'graph' : params.graph = True
        else: params.graph = False

        params.chosen_model = request.form['model']
        if request.form['model'] == 'WAD': params.wad = True
        else: params.wad = False

        return redirect(url_for('data_import'))
    else: 
        return render_template('index.html', models = available_models)

@app.route('/home')
def home():
    return redirect(url_for('index'))

@app.route('/data_import', methods = ('POST','GET'))
def data_import():
    gr = False
    if params.chosen_model == 'GRAM': gr = True

    if request.method ==  'POST':
        X,Y = request.form['input_mat'].strip(), request.form['output_mat'].strip()
        if X == '' or Y == '':
            input_file = request.files['inputsFile']
            outputs_file = request.files['outputsFile']

            if input_file.filename == ''  or outputs_file.filename == '' :
                flash('No data available for computation. Please input data. ')
                return render_template('data_import.html',control = params.graph,gram = gr, wad=params.wad)
            
            input_filename = secure_filename(input_file.filename)
            outputs_filename = secure_filename(input_file.filename)
            
            # bol zadany ako .txt subor
            if not allowed_file(input_filename) or not allowed_file(outputs_filename):
                flash('Only .txt files allowed')
                return render_template('data_import.html',control = params.graph, gram = gr , wad = params.wad)

            input_file.save('FlaskIntro/uploads/vstupy.txt')
            outputs_file.save('FlaskIntro/uploads/vystupy.txt')
            params.inputs_file_path, params.outputs_file_path = 'FlaskIntro/uploads/vstupy.txt', 'FlaskIntro/uploads/vystupy.txt' 
            
            # zle hodnoty pre vahy/p/kontrolovane hodnoty
            if not process_form():
                flash('Parameters weights and nonDiscretionary values should be convertable to type int and parameter p to type float')
                return render_template('data_import.html',control = params.graph, gram = gr , wad = params.wad)
            
            if evaluate_model(): 
                return redirect(url_for('results'))
            # zle zadane matice dat
            else: return render_template('data_import.html',control = params.graph, gram = gr , wad = params.wad)
        else:
            with open('FlaskIntro/uploads/inputs_from_text_area.txt','w') as file:
                file.write(X.replace('\r',''))
            file.close()

            with open('FlaskIntro/uploads/outputs_from_text_area.txt','w') as file:
                file.write(Y.replace('\r',''))
            file.close()
            
            params.inputs_file_path, params.outputs_file_path = 'FlaskIntro/uploads/inputs_from_text_area.txt', 'FlaskIntro/uploads/outputs_from_text_area.txt'

            # zle hodnoty pre vahy/p/kontrolovane hodnoty
            if not process_form():
                flash('Parameters weights (WAD) and nonDiscretionary values should be convertable to type int and parameter p (GRAM) to type float')
                return render_template('data_import.html',control = params.graph, gram = gr , wad = params.wad)
            
            if evaluate_model(): 
                return redirect(url_for('results'))
            # zle zadane matice dat
            else: return render_template('data_import.html',control = params.graph, gram = gr , wad = params.wad)
    else: 
        return render_template('data_import.html',control = params.graph, gram = gr , wad = params.wad)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() == 'txt'

def process_form():
    try :
        if params.graph : 
            if request.form['c_i'].strip() != '':
                params.check_inputs = np.array(list(map(lambda x : int(x), request.form['c_i'].strip().split(' '))))
            if request.form['c_o'].strip() != '':
                params.check_outputs = np.array(list(map(lambda x : int(x), request.form['c_o'].strip().split(' '))))
        if params.wad:
            if request.form['w_i'].strip() != '':
                params.weights_inputs = np.array(list(map(lambda x : int(x), request.form['w_i'].strip().split(' '))))
            if request.form['w_o'].strip() != '':
                params.weights_outputs = np.array(list(map(lambda x : int(x), request.form['w_o'].strip().split(' '))))
        if params.chosen_model == 'GRAM':
            if request.form['p'].strip() != '':
                params.p = float(request.form['p'].strip())
        return True
    except ValueError:
        return False

def evaluate_model():
    try:
        if params.wad:
            params.model = available_models[params.chosen_model]['model'](params.inputs_file_path,params.outputs_file_path,
                                                                            params.weights_inputs, params.weights_outputs,
                                                                            params.vrs,params.check_inputs,params.check_outputs)
        elif params.chosen_model == 'GRAM':
            params.model = available_models[params.chosen_model]['model'](params.inputs_file_path,params.outputs_file_path,
                                                                            params.p,
                                                                            params.vrs,params.check_inputs,params.check_outputs)
        else:
            params.model = available_models[params.chosen_model]['model'](params.inputs_file_path,params.outputs_file_path,
                                                                            params.vrs,params.check_inputs,params.check_outputs)
    except Exception as e:
        flash(e)
        return False
    params.model.solve()
    # generate output file
    params.model.writeSolutionsToFile('FlaskIntro/uploads/outputFile.txt')

    # generate images
    images = []
    try:
        params.model.plot_projections(file = 'FlaskIntro/static/images/projections.png')
        images.append(('projections.png','projekcie'))
    except :
        pass
    params.model.plot_effectivity(file = 'FlaskIntro/static/images/efficiency.png')
    images.append(('efficiency.png','efektivita'))
    params.images = images
    return True

@app.route('/results', methods = ['POST', 'GET'])
def results():
    result_dict, images = params.model.solutions, params.images
    if request.method ==  'POST':
        dmu_to_show = request.form['which_DMU']
        if dmu_to_show == '':
            return render_template('results.html', table = result_dict, images = images )
        dmu_to_show = dmu_to_show.split(' ')
        for dmu in dmu_to_show : params.model.plot_DMU_attributes(which=int(dmu), file_path = 'FlaskIntro/static/images/')
        return render_template('results.html', table = result_dict, images = images, dmu_to_show = dmu_to_show)
    
    return render_template('results.html', table = result_dict, images = images )

@app.route('/results/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename, as_attachment=True)
if __name__ == '__main__':
    app.run(debug =True)