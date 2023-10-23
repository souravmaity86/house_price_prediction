# run flask application
from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


app = Flask(__name__)


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_user_data():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        user_inputs = {
            'crim': float(request.form['crim']),
            'zn': float(request.form['zn']),
            'indus': float(request.form['indus']),
            'chas': int(request.form['chas']),
            'nox': float(request.form['nox']),
            'rm': float(request.form['rm']),
            'age': float(request.form['age']),
            'dis': float(request.form['dis']),
            'rad': int(request.form['rad']),
            'tax': float(request.form['tax']),
            'ptratio': float(request.form['ptratio']),
            'b': float(request.form['b']),
            'lstat': float(request.form['lstat'])
        }
        user_data = CustomData(**user_inputs)
        y_test = user_data.get_user_inputs()
        predict_pipeline = PredictPipeline()
        y_pred = predict_pipeline.predict(y_test)
        results = round(y_pred[0], 2)
        return render_template('results.html', final_result=results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
