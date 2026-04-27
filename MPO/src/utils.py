import os
import logging
from glob import glob
from datetime import datetime
import pytz
import openai
import argparse


openai.log = logging.getLogger("openai")
openai.log.setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)


class HTTPFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("HTTP")


def get_pacific_time():
    current_time = datetime.now()
    pacific = pytz.timezone("Asia/Seoul")
    pacific_time = current_time.astimezone(pacific)
    return pacific_time


def create_logger(logging_dir, name="log"):
    """
    Create a logger that writes to a log file and stdout.
    """
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # Create log file path
    log_file_path = os.path.join(logging_dir, f"{name}.log")

    http_filter = HTTPFilter()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{log_file_path}")],
    )
    logger = logging.getLogger("prompt optimization")
    # logging.getLogger("openai").setLevel(logging.CRITICAL)
    # logging.getLogger("datasets").setLevel(logging.CRITICAL)
    for handler in logging.getLogger().handlers:
        handler.addFilter(http_filter)

    logger.log_dir = logging_dir  # Store logging directory as attribute
    return logger


# fmt: off
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

SUPPORTED_TYPES = {
    'video': {'.mp4', '.avi', '.mov', '.mkv'},
    'image': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'},
    'audio': {'.mp3', '.wav', '.aac', '.flac'},
    'molecule' : {}
}

def check_mm_type(path):
    if isinstance(path, dict) and 'smiles' in path:
        # all molecule modality is dictionary and contains 'smiles' key
        return 'molecule'

    ext = os.path.splitext(path)[1].lower()
    
    for media_type, extensions in SUPPORTED_TYPES.items():
        if ext in extensions:
            return media_type

    raise ValueError(f"Unsupported or unknown file type: {path}")