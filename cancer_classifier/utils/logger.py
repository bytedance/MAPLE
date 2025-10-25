# =============================================================================
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# SCRIPT  : logger.py
# PROJECT : MAPLE (Methylation-Anchor Probe for Low Enrichment)
# PURPOSE : Initialize and configure logging for MAPLE pipelines
#
# OVERVIEW:
#   Provides a standardized logger setup for console and file output.
#   Ensures timestamped, leveled messages for tracking progress and debugging
#   across MAPLE pipeline scripts.
#
# INPUTS  :
#   - output_dir : Directory path where run.log will be saved
#
# OUTPUTS :
#   - run.log file in the specified output directory
#   - Console output of INFO-level messages
#
# USAGE   :
#   init_logger("logs/")
#   logging.info("Pipeline started")
#
# AUTHOR  : Liyuan Zhao
# CREATED : 2025-10-10
# UPDATED : 2025-10-10
#
# NOTE    :
#   - Logging level is set to INFO by default.
#   - Messages are printed both to console and saved to run.log.
# =============================================================================


import logging
from pathlib import Path


def init_logger(output_dir):
    """
    Initialize and configure the logger for MAPLE pipeline scripts.

    This function sets up a logger that outputs INFO-level messages to both
    the console and a file named 'run.log' in the specified output directory.
    The log messages include timestamps and log levels for easier debugging
    and pipeline tracking.

    Parameters
    ----------
    output_dir : str or Path
        Path to the directory where 'run.log' will be created.
        If the directory does not exist, ensure it is created beforehand.

    Returns
    -------
    None
        Configures the root logger in-place.
    """

    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)

    # Create the directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,  # Capture INFO and higher-level messages
        format="%(asctime)s [%(levelname)s] %(message)s",  # Include timestamp and log level
        handlers=[
            logging.FileHandler(output_dir / "run.log"),  # Log to file
            logging.StreamHandler(),  # Log to console
        ],
    )

    logging.info(f"Logger initialized. Log file: {output_dir / 'run.log'}")
