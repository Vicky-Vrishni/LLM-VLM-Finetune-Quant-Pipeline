"""
Logger Utility
--------------
Ye module poore project ke liye ek consistent, professional logging setup
provide karta hai. Plain print() statements ke bajaye, ye:
  - Timestamps ke saath logs print karta hai
  - Log levels support karta hai (INFO, WARNING, ERROR, DEBUG)
  - Logs ko file me bhi save karta hai (debugging ke liye useful)
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Ek configured logger object return karta hai.

    Args:
        name: Logger ka naam (generally __name__ pass karte hain calling file se).
        log_dir: Folder jaha log files save hongi.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Agar logger already configured hai, dobara handlers add na karo
    # (warna duplicate logs print honge)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Log format: [TIME] [LEVEL] [MODULE] Message
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler - terminal me print karega
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler - logs ko file me bhi save karega
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"run_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    # Quick test
    test_logger = get_logger("test_logger")
    test_logger.info("Logger sahi se kaam kar raha hai!")
    test_logger.warning("Yeh ek warning message hai.")
    test_logger.error("Yeh ek error message hai.")