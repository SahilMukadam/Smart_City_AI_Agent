# Smart City AI Agent - Launch Script
# Starts both the FastAPI backend and Streamlit frontend.
# Run from the project root: .\scripts\launch.ps1

Write-Host "🏙️ Smart City AI Agent - Launcher" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Check if venv is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  Virtual environment not activated." -ForegroundColor Yellow
    Write-Host "   Run: venv\Scripts\activate" -ForegroundColor Yellow
    Write-Host ""
}

# Start FastAPI backend in background
Write-Host "🔧 Starting FastAPI backend on port 8000..." -ForegroundColor Green
$backend = Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000" -PassThru -WindowStyle Normal
Write-Host "   PID: $($backend.Id)" -ForegroundColor Gray

# Wait for backend to start
Write-Host "   Waiting for backend..." -ForegroundColor Gray
Start-Sleep -Seconds 3

# Start Streamlit frontend
Write-Host "🎨 Starting Streamlit frontend on port 8501..." -ForegroundColor Green
Write-Host ""
Write-Host "Open in browser: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop both servers." -ForegroundColor Yellow
Write-Host ""

streamlit run frontend/app.py --server.port 8501
