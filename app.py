from flask import Flask, render_template, request, url_for
# import model
import pickle
import numpy as np
import flask

app = Flask(__name__)


@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

def pred_dis(d):
    data = np.array(d).reshape(1, 9)
    clf = pickle.load(open("Heart Disease Logreg Model.pkl", "rb"))
    pred = clf.predict(data)

    return pred

@app.route("/result", methods = ["GET", "POST"])
def hello():
    if request.method == "POST":
        data = request.form.to_dict()
        data = list(data.values())
        data = list(map(int, data))
        result = pred_dis(data)

        if int(result) == 1:
            pred = "The patient has heart disease."
        else:
            pred = "The patient does not have heart disease."

        return render_template("result.html", pred=pred)


# @app.route("/", methods=["POST"])
# def hello():

#      # From html to .py 
#     if request.method == "POST":
#         # age = request.form["age"]
#         # cp = request.form["cp"]
#         # trestbps = request.form["trestbps"]
#         # chol = request.form["chol"]
#         # thalach = request.form["thalach"]
#         # exang = request.form["exang"]
#         # oldpeak = request.form["oldpeak"]
#         # ca = request.form["ca"]
#         # thal = request.form["thal"]

#         # data = [age, cpm, trestbps, chol, thalach, exang, oldpeak, ca, thal]
#         data = request.form.to_dict()
#         data = list(data.values())
#         data = list(map(int, data))

#         dis = pred_dis(data)
#         if int(dis) == 1:
#             pred = "You have heart disease."
#         else:
#             pred = "You do not have heart disease"
#         # global dis
#     # From .py to html
#         return render_template("index.html", pred=pred)

if __name__ == "__main__":
    app.run(debug=True)
