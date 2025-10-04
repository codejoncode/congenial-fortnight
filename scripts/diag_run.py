import sys, os
sys.path.insert(0, os.getcwd())

from scripts.forecasting import HybridPriceForecastingEnsemble

out_lines = []
try:
    out_lines.append('Instantiating HybridPriceForecastingEnsemble for EURUSD')
    fs = HybridPriceForecastingEnsemble('EURUSD')
    out_lines.append('Methods present:')
    for m in ['_load_daily_price_file','_load_intraday_data','_build_intraday_context','_prepare_features','_engineer_features','_get_cross_pair']:
        out_lines.append(f'  {m}: {hasattr(fs, m)}')

    for name in ('intraday_data','monthly_data','price_data'):
        df = getattr(fs, name, None)
        if df is None:
            out_lines.append(f'{name} is None')
            continue
        try:
            out_lines.append(f"{name}: empty={df.empty} shape={getattr(df,'shape',None)} cols={list(df.columns)[:20]}")
            if not df.empty:
                out_lines.append('Sample rows:')
                out_lines.append(df.head(3).to_string())
        except Exception as e:
            out_lines.append(f'Error inspecting {name}: {e}')

    out_lines.append('\nAttempting to run _prepare_features()...')
    try:
        feat = fs._prepare_features()
        if feat is None:
            out_lines.append('Prepared features: None')
        else:
            out_lines.append(f'Prepared features empty={feat.empty} shape={getattr(feat,\'shape\',None)}')
            if not feat.empty:
                out_lines.append('Feature columns sample:')
                out_lines.append(str(list(feat.columns)[:40]))
    except Exception as e:
        out_lines.append(f'Error while preparing features: {e}')

except Exception as e:
    out_lines.append(f'Fatal error instantiating forecasting system: {e}')

with open('/tmp/diag_result.txt','w') as f:
    f.write('\n'.join(out_lines))

print('Diagnostic written to /tmp/diag_result.txt')
