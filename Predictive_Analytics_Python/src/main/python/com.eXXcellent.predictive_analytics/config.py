class Config(object):
    PORT = 8081
    HOST = '0.0.0.0'

    PATH_TO_ROOT = '../../../../'

    UPLOAD_FOLDER = PATH_TO_ROOT + 'files'
    MODEL_FOLDER = PATH_TO_ROOT + 'trained_models'
    LOG_FILE = PATH_TO_ROOT + 'predictive_analytics.log'

    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # set max limit to 50 MB

    SQLALCHEMY_DATABASE_URI = 'sqlite:///db/database.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MODEL_FILES_EXTENSIONS = {'h5', 'pkl', 'json'}
    DATA_FILES_EXTENSIONS = {'csv'}

    DATE_FORMAT = '%d.%m.%Y %H:%M:%S'


class ProductionConfig(Config):
    DEBUG = False


class DevelopmentConfig(Config):
    DEBUG = True
