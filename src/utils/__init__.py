from utils.logger import LoggingManager
import datetime
import hashlib

def setup_logger(pipeline_id):
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    log_file_id = f"pipeline_{pipeline_id}_{timestamp}"
    LoggingManager.setup_logging(log_file_id)

def hash_file(path, bits=64):
    hasher = hashlib.blake2s(digest_size=bits // 8)
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()