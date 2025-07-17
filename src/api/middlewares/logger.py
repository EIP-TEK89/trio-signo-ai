import logging

from flask import Flask, Request

from src.api.logger import *

class LoggerMiddleware:
    def __init__(self, app, logger: logging.Logger):
        self.app: Flask = app
        self.logger: logging.Logger = logger

    def __call__(self, environ, start_response):
        request = Request(environ)
        log_request(self.logger, request)
        log_with_uuid(self.logger, request.environ["uuid"], f"Query params: {request.args}")
        log_with_uuid_debug(self.logger, request.environ["uuid"], f"Header: {request.headers}")

        return self.app(environ, start_response)
