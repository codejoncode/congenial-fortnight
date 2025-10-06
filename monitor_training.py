#!/usr/bin/env python3
"""
Training Monitor - Shows live training progress
"""
import time
import os
from pathlib import Path

log_file = Path('logs/training_output.log')

print("🔍 Training Monitor - Live Progress")
print("="*80)
print("Press Ctrl+C to exit monitor (training will continue)\n")

last_size = 0
if log_file.exists():
    with open(log_file, 'r') as f:
        lines = f.readlines()
        # Show last 30 lines
        for line in lines[-30:]:
            print(line.rstrip())
        last_size = f.tell()

print("\n" + "="*80)
print("Monitoring for new updates...")
print("="*80 + "\n")

try:
    while True:
        if log_file.exists():
            with open(log_file, 'r') as f:
                f.seek(last_size)
                new_lines = f.readlines()
                if new_lines:
                    for line in new_lines:
                        print(line.rstrip())
                last_size = f.tell()
        time.sleep(2)
except KeyboardInterrupt:
    print("\n\n👋 Monitor stopped. Training continues in background.")
