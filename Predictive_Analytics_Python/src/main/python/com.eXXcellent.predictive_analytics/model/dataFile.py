from datetime import datetime

from db.database import db


class DataFile(db.Model):
    """
    model object to store file information in the db
    """

    id = db.Column(db.Integer, primary_key=True)
    filePath = db.Column(db.String, nullable=False)
    dateCreated = db.Column(db.DateTime, default=datetime.utcnow)

    # separator is the character used in the csv file to separate the different columns (often ; , )
    separator = db.Column(db.String(1), nullable=False, default=';')

    def __repr__(self):
        return f"Data File(id={self.id}, file path={self.filePath}, separator={self.separator})"
