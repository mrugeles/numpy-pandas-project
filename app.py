import os
import json
import pandas as pd

from flask import Flask
from flask import render_template, request
from flask import redirect
from flask import jsonify
from flask_restful import Resource, Api
from werkzeug.utils import secure_filename
from data_bot import DataBot
from project import Project
from dataset_attributes import DataSetAtrributes
from Model import Model
from joblib import load

UPLOAD_FOLDER = 'uploads/'
PROJECTS_FOLDER = 'static/projects/'
ALLOWED_EXTENSIONS = set(['csv', 'json'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
api = Api(app)


class Predict(Resource):

    def post(self):
        json_request = request.get_json()

        data = json_request['data']
        for key in data.keys():
            data[key] = [data[key]]
        project = Project()
        project_info = project.get(json_request['project_name'])
        project_info = project_info.to_dict(orient='records')[0]
        model = load(f"{project_info['project_path']}/model.joblib")
        data = pd.DataFrame(data)

        dataBot = DataBot(dataset=data, project_path=project_info['project_path'])
        datasetAttributes = DataSetAtrributes(project_info['project_path'])
        datasetAttributes.load()
        dataBot.pre_process_prediction(datasetAttributes.parameters)
        prediction = list(model.predict(dataBot.features))
        prediction = str(prediction[0])
        print(prediction)
        return {'prediction': prediction}


api.add_resource(Predict, '/predict')

def load_dataset(path):
    """ Load a dataset from the given path. The path can have .csv extension or .json extension.

    :param path:
    :return:
    """
    pass

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/prepare_dataset', methods=['GET', 'POST'])
def prepare_dataset():
    records = {}
    columns = []
    columns_types = {}
    dataset_path = ''
    if request.method == 'POST':
        if 'file' not in request.files:
            error = 'No file part'
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            error = 'No selected file'
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            dataset_path = 'static/' + file_path
            file.save(dataset_path)

            df = pd.read_csv('static/' + file_path).head(3)
            columns_types = df.dtypes.to_dict()
            for columns_type in columns_types:
                columns_types[columns_type] = columns_types[columns_type].name

            columns = df.columns.values
            records = df.to_dict(orient='records')

    return render_template(
        'prepare_dataset.html',
        columns=columns,
        records=records,
        columns_types=columns_types,
        dataset_path=dataset_path
    )


@app.route('/create_model', methods=['GET', 'POST'])
def create_model():
    dataset = None
    dataset_processed = None
    models = None
    scores = None
    best_model = None
    if request.method == 'POST':
        print(request.form)
        project_path = f'{PROJECTS_FOLDER}{request.form.get("project_name")}'
        if not os.path.exists(project_path):
            os.mkdir(project_path)
        dataset = pd.read_csv(request.form.get('dataset_path'))
        columns_types = [key for key in request.form.keys() if '_type' in key]
        for column in columns_types:
            col_data = column.split('_')
            dataset[col_data[0]] = dataset[col_data[0]].astype(request.form.get(column))
        dataset.to_csv(f'{project_path}/dataset.csv', index=False)
        dataBot = DataBot(dataset=dataset,
                          project_path=project_path,
                          target_name=request.form.get('target'),
                          null_threshold=float(request.form.get('null_threshold')) / 100,
                          cardinal_threshold=float(request.form.get('cardinal_threshold')) / 100)
        dataBot.pre_process()

        dataset_processed = dataBot.get_dataset()
        dataset_processed.to_csv(f'{project_path}/dataset_processed.csv', index=False)

        model = Model(dataset_processed, request.form.get('target'))
        model.train_models()
        best_model = model.save_best_model(f'{project_path}/model.joblib')
        models = list(model.training_results['learner'].values)
        scores = list(model.training_results['test_score'].values)

        project_info = {
            'project_name': [request.form.get("project_name")],
            'project_path': [project_path],
            'model_name': [best_model.learner.__class__.__name__],
            'model_score': [best_model.test_score],
            'target': [request.form.get("target")],
            'null_threshold': [request.form.get("null_threshold")],
            'cardinal_threshold': [request.form.get("cardinal_threshold")]
        }

        project = Project(project_info)
        project.save()

    return render_template(
        'model_info.html',
        dataset=dataset.head(3),
        dataset_processed=dataset_processed.head(),
        models=models,
        scores=scores)


@app.route('/list_models', methods=['GET'])
def list_models():
    project = Project()
    models = project.get_projects()
    return render_template(
        'list_models.html',
        models=models)


@app.route('/view_model', methods=['GET'])
def view_model():
    print(request.args.get('project_name'))
    project = Project()
    print(project.get(request.args.get('project_name')))
    return render_template(
        'model_info.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
