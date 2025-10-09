# Gmail App Password Setup Guide

## Why App Password is Needed

Gmail blocks "less secure apps" from using regular passwords. For automated systems like our notification service, you need to generate a special "App Password."

## Setup Steps

### Step 1: Enable 2-Step Verification

1. Go to your Google Account: https://myaccount.google.com/
2. Click **Security** in the left menu
3. Under "How you sign in to Google", click **2-Step Verification**
4. Follow the prompts to enable it (you'll need your phone)

### Step 2: Generate App Password

1. Still in **Security** settings
2. Click **2-Step Verification** (you should see it's now ON)
3. Scroll down and click **App passwords**
4. You may need to sign in again
5. Under "Select app", choose **Mail**
6. Under "Select device", choose **Other (Custom name)**
7. Type: **Trading Signal System**
8. Click **GENERATE**
9. Google will show you a 16-character password like: `abcd efgh ijkl mnop`

### Step 3: Update the Notification Service

**Option A: Set Environment Variable (Recommended)**

```bash
# In your terminal or .bashrc/.zshrc
export EMAIL_PASSWORD="abcdefghijklmnop"  # Remove spaces, use the generated password
```

**Option B: Update the Script Directly**

Edit `scripts/notification_service.py` line 37:

```python
self.email_password = os.getenv('EMAIL_PASSWORD', 'YOUR_16_CHAR_APP_PASSWORD')
```

### Step 4: Test Again

```bash
python scripts/notification_service.py
```

You should see:
```
✅ Email sent successfully to mydecretor@protonmail.com
✅ SMS sent successfully to 7084652230 (T-Mobile)
```

## Troubleshooting

### Still Getting "Username and Password not accepted"?

1. **Double-check the App Password**: Remove all spaces, should be 16 lowercase letters
2. **Verify 2-Step Verification is ON**: Go to Google Account → Security
3. **Use the correct email**: Should be `1man2amazing@gmail.com`
4. **Try generating a new App Password**: Delete old one, create fresh

### Getting "Permission denied" or "Too many login attempts"?

1. Wait 15 minutes
2. Go to https://accounts.google.com/DisplayUnlockCaptcha
3. Click "Continue" to unlock your account
4. Try again

### Want to Use a Different Email Provider?

**ProtonMail Bridge** (if you have paid ProtonMail):
```python
self.smtp_server = 'localhost'
self.smtp_port = 1025
# Use ProtonMail Bridge credentials
```

**Outlook/Office365**:
```python
self.smtp_server = 'smtp.office365.com'
self.smtp_port = 587
self.email_user = 'your_outlook@outlook.com'
self.email_password = 'your_outlook_password'
```

**Yahoo Mail**:
```python
self.smtp_server = 'smtp.mail.yahoo.com'
self.smtp_port = 587
self.email_user = 'your_yahoo@yahoo.com'
self.email_password = 'your_yahoo_app_password'  # Also needs app password
```

## Security Best Practices

### Don't Commit Passwords to Git

**Use Environment Variables**:

1. Create `.env` file (already in .gitignore):
```bash
EMAIL_USER=1man2amazing@gmail.com
EMAIL_PASSWORD=your_app_password_here
NOTIFICATION_EMAIL=mydecretor@protonmail.com
SMS_NUMBER=7084652230
```

2. Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

3. Install python-dotenv:
```bash
pip install python-dotenv
```

### For Production (Cloud Run)

Set environment variables in `cloudbuild.yaml`:
```yaml
env:
  - EMAIL_USER=1man2amazing@gmail.com
  - EMAIL_PASSWORD=${_EMAIL_PASSWORD}  # Set in Google Cloud Console
```

Then set secret in Google Cloud:
```bash
gcloud secrets create email-password --data-file=-
# Paste your app password when prompted
```

## Alternative: Use SendGrid (More Reliable for Production)

SendGrid offers 100 free emails/day:

```python
import sendgrid
from sendgrid.helpers.mail import Mail

sg = sendgrid.SendGridAPIClient(api_key=os.getenv('SENDGRID_API_KEY'))
message = Mail(
    from_email='noreply@yourdomain.com',
    to_emails='mydecretor@protonmail.com',
    subject='Trading Signal',
    html_content='<strong>Your signal details</strong>'
)
response = sg.send(message)
```

Benefits:
- No Gmail restrictions
- Better deliverability
- Email tracking
- Free tier: 100 emails/day

Sign up: https://sendgrid.com/

## Quick Reference

**What You Need**:
1. Gmail account with 2-Step Verification ON
2. Generated App Password (16 characters)
3. Environment variable or script update

**Test Command**:
```bash
python scripts/notification_service.py
```

**Expected Output**:
```
✅ SUCCESS: Notifications are working!
   - Email sent to: mydecretor@protonmail.com
   - SMS sent to: 7084652230 (T-Mobile)
```

---

**Need Help?**
- Gmail App Passwords: https://support.google.com/accounts/answer/185833
- 2-Step Verification: https://support.google.com/accounts/answer/185839
- Less Secure Apps: https://support.google.com/accounts/answer/6010255
