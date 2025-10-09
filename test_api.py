import requests
import time

# Wait for server to start
time.sleep(2)

try:
    response = requests.get('http://127.0.0.1:8001/api/historical/?pair=EURUSD&days=7')
    print('Status:', response.status_code)
    print('Response length:', len(response.text))
    if response.status_code == 200:
        print('Response preview:', response.text[:300])
    else:
        print('Error response:', response.text)
except Exception as e:
    print('Error:', e)