from flask import Flask
from src.logger import logging

app = Flask(__name__)

@app.route('/', methods=['Get', 'POST'])
def index():
    logging.info('Checking Logging System')
    return "ADULT INCOME PRED MODEL WITH COMPLETE ML PIPELINE"


if __name__ == "__main__":
    app.run(debug=True)