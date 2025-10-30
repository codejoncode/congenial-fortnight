#!/usr/bin/env python3
"""
Quick test script for free notifications
Replace YOUR_APP_PASSWORD with your actual Gmail app password
"""

import os
os.environ['GMAIL_USERNAME'] = '1man2amazing@gmail.com'
os.environ['GMAIL_APP_PASSWORD'] = 'YOUR_APP_PASSWORD'  # Replace with your 16-char app password
os.environ['NOTIFICATION_EMAIL'] = 'mydecretor@protonmail.com'
os.environ['NOTIFICATION_SMS'] = '7734921722'

# Import and run the notification test
from free_notifications import test_free_notifications

if __name__ == "__main__":
    print("🔑 Using Gmail account: 1man2amazing@gmail.com")
    print("📧 Sending to: mydecorator@protonmail.com")
    print("📱 Sending to: 7734921722")
    print()
    test_free_notifications()