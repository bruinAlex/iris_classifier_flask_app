from flask import Flask, render_template, request
# from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import irisModel
import ast

app = Flask(__name__)
# api = Api(app)

# create new model object
model = irisModel()

# load trained classifier
clf_path = 'models/SentimentClassifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

# # argument parsing
# parser = reqparse.RequestParser()
# parser.add_argument('query', action='append')


@app.route('/', methods=('GET', 'POST'))
def submit_page():
    return render_template('submit.html')

@app.route('/submit', methods = ['POST', 'GET'])
def echo():
    args = request.form['email']
    args = args.split(",")
    args = list(map(float, args))

    prediction = model.predict(np.array([args]))

    pred_text = (prediction[0])

    return render_template('submit.html', text = pred_text)
    # return render_template('submit.html', text = args)


# class PredictIris(Resource):
#     def post(self):
#     # def get(self):
#         # use parser and find the user's query
#         args = parser.parse_args()
#         # print(type(args['query']), '\n', args['query'])
#         args = args['query'][0].split(",")
#         # args = ast.literal_eval(args)
#         # args = [n.strip() for n in args]
#         # print(args['query'])

#         # vectorize the user's query and make a prediction

#         # uq_vectorized = model.vectorizer_transform(
#         #     np.array([user_query]))

#         prediction = model.predict(np.array([args]))
#         # pred_proba = model.predict_proba(np.array([args['query']]))
#         # prediction = model.predict(np.array([user_query]))
#         # pred_proba = model.predict_proba(np.array([user_query]))

#         # Output 'Negative' or 'Positive' along with the score
#         pred_text = (prediction[0])

#         # if prediction == 0:
#         #     pred_text = 'Negative'
#         # else:
#         #     pred_text = 'Positive'
            
#         # round the predict proba value and set to new variable
#         # confidence = round(pred_proba[0], 3)

#         # create JSON object
#         output = {'prediction': pred_text}
#         # output = {'prediction': pred_text, 'confidence': confidence}
        
#         return output['prediction']

# api.add_resource(Submit, '/')
# api.add_resource(PredictIris, '/pred')
  
# # example of another endpoint
# api.add_resource(PredictRatings, '/ratings')

if __name__ == '__main__':
    app.run(debug=True)