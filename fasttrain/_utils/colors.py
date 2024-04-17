_COLORS = {
    'green': '\033[1;32m{text}\033[0m',
    'orange': '\033[38;5;208m{text}\033[0m',
    'purple': '\033[38;5;092m{text}\033[0m',
    }


def paint(text: str, color: str) -> str:
    assert color in _COLORS
    return _COLORS[color].format(text=text)