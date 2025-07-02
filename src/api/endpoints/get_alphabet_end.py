import ipaddress

from flask import request, jsonify

from src.datasample import *

def get_alphabet_end(sample_history: dict[int, DataSample]):
    try:
        ip: int = ipaddress.ip_address(request.remote_addr)
    except:
        return jsonify({'error': 'Invalid IP address'}), 400

    try:
        del sample_history[ip]
    except:
        pass
    return jsonify({'message': 'Sample history deleted'}), 200
