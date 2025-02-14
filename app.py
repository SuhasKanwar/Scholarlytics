import sys
from flask import Flask, request, render_template
from src.logger import logging as logger
from src.exception import CustomException
from src.pipelines.predict_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    elif request.method == 'POST':
        try:
            logger.info('Prediction started...')
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=int(request.form.get('reading_score')),
                writing_score=int(request.form.get('writing_score'))
            )
            df = data.get_data_as_dataframe()
            logger.info(df)
            pipeline = PredictionPipeline()
            prediction = pipeline.predict(df)
            logger.info(f'Prediction: {prediction}')
            # return render_template('predict.html', prediction=prediction[0])
            return { 'prediction': prediction[0] }
        except Exception as e:
            raise CustomException(e, sys)
    else:
        return 'Invalid Request Method'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)