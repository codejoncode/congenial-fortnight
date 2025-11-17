# Start Django Server
# Activates virtual environment and runs the server

Write-Host "🚀 Starting Django server with virtual environment..." -ForegroundColor Green

# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

# Start server
python manage.py runserver
