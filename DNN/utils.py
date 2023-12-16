import sys
import os
import errno
import logging
import re

class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    WHITE = '\033[37m'
    YELLOW = '\033[33m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    RED = '\033[31m'
    MAGENTA = '\033[35m'
    BLACK = '\033[30m'
    BHEADER = BOLD + '\033[95m'
    BOKBLUE = BOLD + '\033[94m'
    BOKGREEN = BOLD + '\033[92m'
    BWARNING = BOLD + '\033[93m'
    BFAIL = BOLD + '\033[91m'
    BUNDERLINE = BOLD + '\033[4m'
    BWHITE = BOLD + '\033[37m'
    BYELLOW = BOLD + '\033[33m'
    BGREEN = BOLD + '\033[32m'
    BBLUE = BOLD + '\033[34m'
    BCYAN = BOLD + '\033[36m'
    BRED = BOLD + '\033[31m'
    BMAGENTA = BOLD + '\033[35m'
    BBLACK = BOLD + '\033[30m'

    @staticmethod
    def cleared(s):
        return re.sub("\033\[[0-9][0-9]?m", "", s)

def red(message):
    return BColors.RED + str(message) + BColors.ENDC

def b_red(message):
    return BColors.BRED + str(message) + BColors.ENDC

def blue(message):
    return BColors.BLUE + str(message) + BColors.ENDC

def b_yellow(message):
    return BColors.BYELLOW + str(message) + BColors.ENDC

def green(message):
    return BColors.GREEN + str(message) + BColors.ENDC

def b_green(message):
    return BColors.BGREEN + str(message) + BColors.ENDC

def set_logger(out_dir=None, model_type=None):
    console_format = BColors.OKBLUE + '[%(levelname)s]' + BColors.ENDC + ' (%(name)s) %(message)s'
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(console_format))
    logger.addHandler(console)
    if out_dir:
        file_format = '[%(levelname)s] (%(name)s) %(message)s'
        log_file = logging.FileHandler(os.path.join(out_dir, f'{model_type}_log.txt'), mode='w')
        log_file.setLevel(logging.DEBUG)
        log_file.setFormatter(logging.Formatter(file_format))
        logger.addHandler(log_file)

def mkdir_p(path):
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_root_dir():
    return os.path.dirname(sys.argv[0])

def bincounts(array):
    num_rows, num_cols = array.shape[0], array.shape[1] if array.ndim > 1 else 1
    counters = []
    mfe_list = []
    for col in range(num_cols):
        counter = {}
        for row in range(num_rows):
            element = array[row, col]
            if element in counter:
                counter[element] += 1
            else:
                counter[element] = 1
        max_count = 0
        for element in counter:
            if counter[element] > max_count:
                max_count = counter[element]
                mfe = element
        counters.append(counter)
        mfe_list.append(mfe)
    return counters, mfe_list

def ltos(*args):
    outputs = []
    for arg in args:
        if type(arg) == list:
            out = ' '.join(['%.3f' % e for e in arg])
            outputs.append('[' + out + ']' if len(arg) > 1 else out)
        else:
            outputs.append(str(arg))
    return tuple(outputs)

def print_args(args, path=None):
    if path:
        output_file = open(path, 'w')
    logger = logging.getLogger(__name__)
    logger.info("Arguments:")
    args.command = ' '.join(sys.argv)
    items = vars(args)
    for key in sorted(list(items.keys()), key=lambda s: s.lower()):
        value = items[key]
        value = "None" if not value else value
        logger.info(f"  {key}: {value}")
        if path is not None:
            output_file.write(f"  {key}: {value}\n")
    if path:
        output_file.close()
    del args.command
