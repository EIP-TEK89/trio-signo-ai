import logging
from flask import Request
import time
import uuid

def get_endpoint(request: Request) -> str:
    endpoint = ""
    for i in range(len(request.base_url)):
        if len(request.root_url) <= i or request.base_url[i] != request.root_url[i]:
            endpoint += request.base_url[i]
    return endpoint





def setup_logger(debug: bool = False) -> logging.Logger:
    # Create or retrieve a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Check if the logger already has handlers configured
    if not logger.handlers:
        # Create a stream handler with a specific log level
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.DEBUG)  # Set to DEBUG to capture all debug logs
        c_format = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

    # Disable propagation to the root logger
    logger.propagate = False

    return logger

def log_with_uuid(logger: logging.Logger, uuid: uuid.UUID, message: str):
    logger.info(f"[{uuid}] {message}")

def log_with_uuid_debug(logger: logging.Logger, uuid: uuid.UUID, message: str):
    logger.debug(f"[{uuid}] {message}")

def log_with_uuid_warning(logger: logging.Logger, uuid: uuid.UUID, message: str):
    logger.warning(f"[{uuid}] {message}")

def log_request(logger: logging.Logger, request: Request):
    log_with_uuid(logger, request.environ["uuid"], f"{request.method} /{get_endpoint(request)} - {request.remote_addr}")

def log_response(logger: logging.Logger, request: Request, status: str):
    timing = round(time.time() - request.environ["time"], 2)

    log_with_uuid(logger, request.environ["uuid"], f"{request.method} /{get_endpoint(request)} - {status} - {timing} sec")

def log_callback_request(logger: logging.Logger, uuid: uuid.UUID, method: str, url: str):
    log_with_uuid(logger, uuid, f"Callback {method} {url}")

def log_callback_response(logger: logging.Logger, uuid: uuid.UUID, status: int):
    log_with_uuid(logger, uuid, f"Callback response {status}")

def log_callback_response_warning(logger: logging.Logger, uuid: uuid.UUID, status: int, error: str):
    log_with_uuid_warning(logger, uuid, f"Callback response {status} {error}")
