from datetime import datetime

from db.database import db


class PredictionModel(db.Model):
    """
    model object to store the information of trained models in the db
    """

    id = db.Column(db.Integer, primary_key=True)
    # The type of model used (type = name of the model module in '/models' directory)
    modelType = db.Column(db.String(100), nullable=False)
    modelPath = db.Column(db.String(300))
    multivariate = db.Column(db.Boolean, default=False)
    rmse = db.Column(db.Float)
    resolution = db.Column(db.String(3))
    dateCreated = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"Prediction Model(id={self.id}, model type={self.modelType}, model path={self.modelPath})"
