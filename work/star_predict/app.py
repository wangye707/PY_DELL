from flask import  Flask
from .route import test_index

app = Flask(__name__)

app.add_url_rule("/", "test_index", test_index)

def run_app():
    app.run()



