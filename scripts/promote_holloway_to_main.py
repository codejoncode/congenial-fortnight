from pathlib import Path
import shutil
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
SYMBOLS = ['EURUSD','XAUUSD']
MAP = {
    'H1':'h1_complete_holloway',
    'H4':'h4_complete_holloway',
    'Daily':'daily_complete_holloway',
    'Weekly':'weekly_complete_holloway',
    'Monthly':'monthly_complete_holloway',
}

def nonnull_ohlc_count(path):
    try:
        df = pd.read_csv(path, nrows=2000)
    except Exception:
        df = pd.read_csv(path, nrows=2000, engine='python')
    open_nonnull=0
    close_nonnull=0
    for c in df.columns:
        if c.lower()=='open':
            open_nonnull = int(df[c].notna().sum())
        if c.lower()=='close':
            close_nonnull = int(df[c].notna().sum())
    return open_nonnull, close_nonnull

replaced=[]
for sym in SYMBOLS:
    for tf_main, hollow_suffix in MAP.items():
        main_name = f'{sym}_{tf_main}.csv'
        hollow_name = f'{sym}_{hollow_suffix}.csv'
        main_path = DATA / main_name
        hollow_path = DATA / hollow_name
        if not main_path.exists():
            print('skip missing main:', main_path)
            continue
        if not hollow_path.exists():
            print('skip missing holloway:', hollow_path)
            continue
        main_open, main_close = nonnull_ohlc_count(main_path)
        hollow_open, hollow_close = nonnull_ohlc_count(hollow_path)
        print(f'Checking {main_name}: main open/close={main_open}/{main_close}, hollow open/close={hollow_open}/{hollow_close}')
        if (main_open==0 and main_close==0) and (hollow_open>0 or hollow_close>0):
            # backup main
            orig = main_path.with_suffix(main_path.suffix + '.orig')
            if not orig.exists():
                print(' Backing up', main_path.name, '->', orig.name)
                shutil.copy2(main_path, orig)
            # copy hollow to main
            print(' Replacing', main_path.name, 'with', hollow_path.name)
            shutil.copy2(hollow_path, main_path)
            replaced.append((main_name, hollow_name))
        else:
            print(' No replacement needed')

if replaced:
    # run cleaner and metadata refresh
    try:
        print('\nRunning cleaner...')
        subprocess.run([sys.executable, str(ROOT / 'scripts' / 'clean_trailing_commas.py')], check=True)
    except Exception as e:
        print('cleaner error', e)
    try:
        print('\nRefreshing metadata...')
        subprocess.run([sys.executable, str(ROOT / 'scripts' / 'refresh_metadata_for_prices.py')], check=True)
    except Exception as e:
        print('metadata refresh error', e)

print('\nReplaced files:')
for a,b in replaced:
    print(' ', a, '<-', b)

# final scan summary
print('\nFinal main file OHLC counts:')
for sym in SYMBOLS:
    for tf_main in MAP.keys():
        main_name = f'{sym}_{tf_main}.csv'
        main_path = DATA / main_name
        if main_path.exists():
            o,c = nonnull_ohlc_count(main_path)
            print(' ', main_name, 'open/close:', o, c)
