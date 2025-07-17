import logging

from werkzeug.wrappers import Request

from src.api.logger import *


class AfterLoggerMiddleware:
    def __init__(self, app, logger: logging.Logger):
        self.app = app
        self.logger = logger

    def __call__(self, environ, start_response):
        # Helper function to capture status and headers
        resp_status = ["-1 UNKNOWN"]

        def custom_start_response(status, headers, exc_info=None):
            # Store status and headers to use later
            resp_status[0] = status
            return start_response(status, headers, exc_info)

        # Call the Flask application with the custom start_response
        response = self.app(environ, custom_start_response)
        log_response(self.logger, Request(environ), resp_status[0])

        return response
