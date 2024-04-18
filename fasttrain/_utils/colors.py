_COLORS = {
    'green': '\033[1;32m{text}\033[0m',
    'orange': '\033[38;5;208m{text}\033[0m',
    'purple': '\033[38;5;092m{text}\033[0m',
    }

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def paint(text: str, color: str) -> str:
    assert color in _COLORS
    return _COLORS[color].format(text=text)


def warning(text):
    return f'{bcolors.WARNING}{text}{bcolors.ENDC}'


def success(text):
    return f'{bcolors.OKGREEN}{text}{bcolors.ENDC}'


def fail(text):
    return f'{bcolors.FAIL}{text}{bcolors.ENDC}'


def blue(text):
    return f'{bcolors.OKBLUE}{text}{bcolors.ENDC}'


def underline(text):
    return f'{bcolors.UNDERLINE}{text}{bcolors.ENDC}'