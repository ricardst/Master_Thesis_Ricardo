import re

label_time_shift_str = '-2h 0min 21s'
shift_match = re.match(r'(?:(-?\d+)h)?\s*(?:(-?\d+)min)?\s*(?:(-?\d+)s)?', label_time_shift_str)
if shift_match:
    shift_hours = int(shift_match.group(1) or 0); shift_minutes = int(shift_match.group(2) or 0); shift_seconds = int(shift_match.group(3) or 0)
    total_shift_seconds = (shift_hours * 3600) + (shift_minutes * 60) + shift_seconds
    print(total_shift_seconds)        