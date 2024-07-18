import logging
import logging.config
from pathlib import Path

from ..utils import read_json

this_file_path = Path(__file__).parent


class ColorFormatter(logging.Formatter):
    def __init__(self, format: str, *args, **kwargs):
        blue = "\x1b[94;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        reset = "\x1b[0m"

        self.formats = {
            logging.DEBUG: blue + format + reset,
            logging.INFO: format,
            logging.WARNING: yellow + format + reset,
            logging.ERROR: red + format + reset,
        }

    def format(self, record):
        log_fmt = self.formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


def setup_logging(
    save_dir,
    log_config=this_file_path.joinpath("logger_config.json"),
    level=logging.INFO,
):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        config["handlers"]["console"]["level"] = level
        config["handlers"]["info_file_handler"]["filename"] = str(
            save_dir / config["handlers"]["info_file_handler"]["filename"]
        )
        logging.config.dictConfig(config)
    else:
        print(
            "Warning: logging configuration file is not found in {}.".format(log_config)
        )
        logging.basicConfig(level=level)
