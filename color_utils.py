# color_utils.py

import sys
sys.stdout.reconfigure(encoding='utf-8')


def get_color_by_magnitude(magnitude):
    """Deprem şiddetine göre gösterge renginin ayarlanması"""
    if magnitude >= 5: return 'red'
    elif magnitude >= 4: return 'orange'
    elif magnitude >= 3: return 'yellow'
    else: return 'green'
