import numpy as np
from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pickle

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///data.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

model = pickle.load(open("model.pkl", "rb"))

class Data(db.Model):
    sno = db.Column(db.Integer, primary_key = True)
    Sepal_Length = db.Column(db.REAL, nullable = False)
    Sepal_Width = db.Column(db.REAL, nullable = False)
    Petal_Length = db.Column(db.REAL, nullable = False)
    Petal_Width = db.Column(db.REAL, nullable = False)
    # Result = db.Column(db.String(100), nullable=False)
    date_created = db.Column(db.DateTime, default = datetime.now)

    def __repr__(self) -> int:
        return f"{self.sno}"

@app.route("/", methods = ['GET','POST'])
def home():
    return render_template("index1.html", allData=Data.query.all())

@app.route("/predict", methods = ["POST","GET"])
def predict():
    if request.method == "POST":
        Sepal_Length = request.form['sl']
        Sepal_Width = request.form['sw']
        Petal_Length = request.form['pl']
        Petal_Width = request.form['pw']
        data = Data(Sepal_Length=Sepal_Length, Sepal_Width=Sepal_Width, Petal_Length=Petal_Length, Petal_Width=Petal_Width)
        db.session.add(data)
        db.session.commit()
    allData = Data.query.all()
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    # Result = prediction[0]
    # data = Data(Result=prediction[0])
    # db.session.add(data)
    # db.session.commit()
    # allData = Data.query.all()
    return render_template("index1.html", prediction_text = "The flower species is {}".format(prediction[0]),allData=allData)

if __name__ == "__main__":
    app.run(debug=True, port=8888)