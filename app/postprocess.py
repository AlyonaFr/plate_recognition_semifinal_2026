import re
from .constants import LETTERS, DIGITS

LETTER_LIKE = {
    '0':'O', '8':'B', '4':'A', '7':'T', '1':'T', '5':'C',
}
DIGIT_LIKE = {
    'O':'0', 'B':'8', 'T':'7', 'A':'4', 'C':'0', 'Y':'4', 'X':'8'
}


def normalize_symbol(ch: str, expected: str) -> str:
    ch = ch.upper()
    if expected == 'L':
        if ch in LETTERS: return ch
        return LETTER_LIKE.get(ch, '#')
    if expected == 'D':
        if ch in DIGITS: return ch
        return DIGIT_LIKE.get(ch, '#')
    return '#'


def enforce_plate_mask(text: str) -> str:
    text = ''.join(c for c in text.upper() if c.isalnum() or c == '#')
    # accepted lengths 8 or 9: LDDDLL + 2/3 region digits
    if len(text) not in (8, 9):
        return text[:9].ljust(8, '#') if text else '########'
    mask = 'LDDDLL' + ('DD' if len(text) == 8 else 'DDD')
    chars = [normalize_symbol(c, m) if c != '#' else '#' for c, m in zip(text, mask)]
    if len(chars) == 9 and chars[6] not in ('1','2','7','#'):
        chars[6] = '#'
    return ''.join(chars)


def is_valid_plate(text: str) -> bool:
    return bool(re.fullmatch(r'[ABEKMHOPCTYX][0-9]{3}[ABEKMHOPCTYX]{2}[0-9]{2,3}', text))
