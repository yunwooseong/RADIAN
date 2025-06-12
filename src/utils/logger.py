# coding: utf-8
# @email: enoche.chow@gmail.com

"""
###############################
"""

import logging
import os

from utils.utils import log_directory


def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
    """
    LOGROOT = log_directory(config)
    dir_name = os.path.dirname(LOGROOT)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    logfilename = "{}-{}.log".format(config["model"], config["dataset"])

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "[%(asctime)s] %(message)s"
    filedatefmt = "%Y.%m.%d %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "[%(asctime)s] %(message)s"
    sdatefmt = "%Y.%m.%d %H:%M:%S"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    if config["state"] is None or config["state"].lower() == "info":
        level = logging.INFO
    elif config["state"].lower() == "debug":
        level = logging.DEBUG
    elif config["state"].lower() == "error":
        level = logging.ERROR
    elif config["state"].lower() == "warning":
        level = logging.WARNING
    elif config["state"].lower() == "critical":
        level = logging.CRITICAL
    else:
        level = logging.INFO
    # comment following 3 lines and handlers = [sh, fh] to cancel file dump.
    fh = logging.FileHandler(logfilepath, "w", "utf-8")
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(
        level=level,
        # handlers=[sh]
        handlers=[sh, fh],
    )

    copy_model_file(config, LOGROOT)


def copy_model_file(config, log_dir):
    """copy model file to log directory"""
    import shutil

    try:
        model_name = config["model"]

        model_file_paths = []

        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        model_file = os.path.join(models_dir, f"{model_name.lower()}.py")
        if os.path.exists(model_file):
            model_file_paths.append(model_file)

        if os.path.exists(model_file):
            # copy model file to log directory
            file_name = os.path.basename(model_file)
            dest_path = os.path.join(log_dir, f"{file_name}")
            shutil.copy2(model_file, dest_path)

    except Exception as e:
        print(f"Error copying model file: {e}")
