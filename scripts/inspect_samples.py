from pathlib import Path

targets = [
    'data/EURUSD_H1.csv',
    'data/EURUSD_H4.csv',
    'data/XAUUSD_Daily.csv',
]

for tf in targets:
    p = Path(tf)
    print('\n' + '='*80)
    print('FILE:', p)
    if not p.exists():
        print('  MISSING')
        continue
    try:
        with p.open('r', encoding='utf-8', errors='replace') as fh:
            for i, line in enumerate(fh, start=1):
                if i == 1:
                    print('HEADER:', repr(line))
                else:
                    print(f'{i:03}:', repr(line))
                if i >= 40:
                    break
    except Exception as e:
        print('  ERROR reading:', e)
