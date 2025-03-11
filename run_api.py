import argparse
import logging

from flask import Flask
from flask_cors import CORS
from waitress import serve

from src.logger import setup_logger

from src.api.middlewares.after_logger import AfterLoggerMiddleware
from src.api.middlewares.logger import LoggerMiddleware
from src.api.middlewares.init import InitMiddleware

from src.api.endpoints.ping import ping
from src.api.endpoints.get_alphabet import get_alphabet
from src.api.endpoints.get_alphabet2 import get_alphabet2
from src.api.endpoints.get_alphabet_end import get_alphabet_end
from src.api.endpoints.get_sign_recognizer_model import get_sign_recognizer_model

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import *

from src.model_class.transformer_sign_recognizer import *
from run_model import load_hand_landmarker

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--port',
        help='Port the Flask app will listen on.',
        required=False,
        default=5000)
    parser.add_argument(
        '--debug',
        help='To run Flask app in debug mode.',
        required=False,
        default=False)
    parser.add_argument(
        '--model',
        help='Pick model.',
        required=False,
        default="alphabet")

    args = parser.parse_args()
    port = int(args.port)

    # Setup logger
    logger: logging.Logger = setup_logger(args.debug)
    logger.debug(f"Logger setup")

    hand_tracker: HandLandmarker = load_hand_landmarker(1)
    alphabet_recognizer: SignRecognizerTransformer = SignRecognizerTransformer.loadModelFromDir(args.model)

    logger.debug(f"AI Model loaded successfully")

    # Setup Flask app
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})  # Permet toutes les origines




    # Adding middleware
    app.wsgi_app = AfterLoggerMiddleware(app.wsgi_app, logger)
    # LoggerMiddleware will log every request
    app.wsgi_app = LoggerMiddleware(app.wsgi_app, logger)

    app.wsgi_app = InitMiddleware(app.wsgi_app)
    logger.debug("Middleware setup")




    # Endpoints
    app.add_url_rule('/ping', view_func=ping, methods=['GET'])
    datasamples_instance: dict[int, DataSample2] = {}
    app.add_url_rule('/get-alphabet', view_func=get_alphabet, methods=['POST'],
                     defaults={"hand_tracker": hand_tracker,
                               "alphabet_recognizer": alphabet_recognizer,
                               "sample_history": datasamples_instance})
    app.add_url_rule('/get-alphabet2', view_func=get_alphabet2, methods=['POST'],
                     defaults={"alphabet_recognizer": alphabet_recognizer,
                               "sample_history": datasamples_instance})
    app.add_url_rule('/get-alphabet-end', view_func=get_alphabet_end, methods=['DELETE'],
                     defaults={"sample_history": datasamples_instance})

    app.add_url_rule('/get-sign-recognizer-model/<model_name>', view_func=get_sign_recognizer_model, methods=['GET'])


    if args.debug:
        print(f"Running dev server: {port}")
        app.run(port=port, debug=True)
    else:
        print(f"Running production server: {port}")
        serve(app, port=port, threads=16)


if __name__ == '__main__':
  main()
