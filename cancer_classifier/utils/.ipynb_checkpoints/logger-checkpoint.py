import logging
from pathlib import Path

def init_logger(output_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(Path(output_dir).joinpath("run.log")),
            logging.StreamHandler()
        ]
    )