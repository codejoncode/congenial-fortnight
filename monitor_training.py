#!/usr/bin/env python3#!/usr/bin/env python3

""""""

Training Progress MonitorTraining Monitor - Shows live training progress

Real-time monitoring of training progress"""

"""import time

import osimport os

import timefrom pathlib import Path

from pathlib import Path

from datetime import datetime, timedeltalog_file = Path('logs/training_output.log')



def get_latest_log():print("🔍 Training Monitor - Live Progress")

    """Get the most recent training log file"""print("="*80)

    logs = list(Path('.').glob('training_run_*.log'))print("Press Ctrl+C to exit monitor (training will continue)\n")

    if not logs:

        return Nonelast_size = 0

    return max(logs, key=os.path.getmtime)if log_file.exists():

    with open(log_file, 'r') as f:

def tail_log(log_file, n=20):        lines = f.readlines()

    """Get last N lines of log file"""        # Show last 30 lines

    try:        for line in lines[-30:]:

        with open(log_file, 'r') as f:            print(line.rstrip())

            lines = f.readlines()        last_size = f.tell()

            return lines[-n:]

    except:print("\n" + "="*80)

        return []print("Monitoring for new updates...")

print("="*80 + "\n")

def parse_progress(log_file):

    """Parse training progress from log"""try:

    try:    while True:

        with open(log_file, 'r') as f:        if log_file.exists():

            content = f.read()            with open(log_file, 'r') as f:

                        f.seek(last_size)

        info = {                new_lines = f.readlines()

            'started': None,                if new_lines:

            'current_pair': None,                    for line in new_lines:

            'feature_engineering': False,                        print(line.rstrip())

            'training': False,                last_size = f.tell()

            'completed': False,        time.sleep(2)

            'total_features': None,except KeyboardInterrupt:

            'fundamental_features': None,    print("\n\n👋 Monitor stopped. Training continues in background.")

            'accuracy': None,
        }
        
        # Check for start time
        if '--- Processing EURUSD ---' in content:
            info['started'] = True
            info['current_pair'] = 'EURUSD'
        if '--- Processing XAUUSD ---' in content:
            info['current_pair'] = 'XAUUSD'
        
        # Check for feature engineering
        if 'Engineered' in content and 'features from' in content:
            info['feature_engineering'] = True
            # Extract feature count
            for line in content.split('\n'):
                if 'Engineered' in line and 'features' in line:
                    parts = line.split('Engineered')[1].split('features')[0].strip()
                    try:
                        info['total_features'] = int(parts)
                    except:
                        pass
        
        # Check for fundamental features
        if 'fundamental features' in content:
            for line in content.split('\n'):
                if 'Fundamental features:' in line:
                    try:
                        info['fundamental_features'] = int(line.split(':')[1].strip())
                    except:
                        pass
        
        # Check for training
        if 'Training until iteration' in content or 'Early stopping' in content:
            info['training'] = True
        
        # Check for completion
        if 'Training completed' in content or 'TRAINING COMPLETE' in content:
            info['completed'] = True
        
        # Check for accuracy
        if 'Accuracy:' in content:
            for line in content.split('\n'):
                if 'Accuracy:' in line:
                    try:
                        acc = line.split('Accuracy:')[1].strip().split()[0]
                        info['accuracy'] = float(acc)
                    except:
                        pass
        
        return info
    except:
        return None

def format_time_elapsed(start_time):
    """Format elapsed time"""
    elapsed = datetime.now() - start_time
    minutes = int(elapsed.total_seconds() / 60)
    seconds = int(elapsed.total_seconds() % 60)
    return f"{minutes}m {seconds}s"

def main():
    """Monitor training progress"""
    print("\n" + "="*80)
    print("TRAINING PROGRESS MONITOR")
    print("="*80)
    
    log_file = get_latest_log()
    if not log_file:
        print("\n❌ No training log found. Is training running?")
        print("   Start training with: python -m scripts.automated_training")
        return
    
    print(f"\n📄 Monitoring: {log_file}")
    print(f"   Started: {datetime.fromtimestamp(os.path.getmtime(log_file)).strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = datetime.fromtimestamp(os.path.getmtime(log_file))
    
    try:
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            
            print("\n" + "="*80)
            print("TRAINING PROGRESS MONITOR")
            print("="*80)
            
            # Parse progress
            info = parse_progress(log_file)
            if info:
                print(f"\n⏱️  Elapsed Time: {format_time_elapsed(start_time)}")
                
                if info['current_pair']:
                    print(f"📊 Current Pair: {info['current_pair']}")
                
                if info['total_features']:
                    print(f"🔧 Features Engineered: {info['total_features']}")
                
                if info['fundamental_features']:
                    print(f"📈 Fundamental Features: {info['fundamental_features']}")
                
                # Progress indicators
                steps = []
                if info['started']:
                    steps.append("✅ Started")
                if info['feature_engineering']:
                    steps.append("✅ Feature Engineering")
                if info['training']:
                    steps.append("🔄 Training Model...")
                if info['completed']:
                    steps.append("✅ Completed!")
                
                if steps:
                    print(f"\n📋 Progress: {' → '.join(steps)}")
                
                if info['accuracy']:
                    print(f"\n🎯 Latest Accuracy: {info['accuracy']:.4f}")
            
            # Show recent log lines
            print("\n" + "-"*80)
            print("RECENT LOG OUTPUT:")
            print("-"*80)
            
            recent_lines = tail_log(log_file, 15)
            for line in recent_lines:
                print(line.rstrip())
            
            print("\n" + "-"*80)
            print("Press Ctrl+C to stop monitoring (training will continue)")
            print("-"*80)
            
            # Check if training is still running
            if info and info['completed']:
                print("\n🎉 Training completed!")
                break
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped. Training continues in background.")
        print(f"   Log file: {log_file}")
        print(f"   Check progress with: tail -f {log_file}")

if __name__ == "__main__":
    main()
