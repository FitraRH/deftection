# Laravel + Flask + nginx Integration

A unified web application combining Laravel for web functionality and Flask for AI services, orchestrated through nginx reverse proxy.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Client      │────│     nginx       │────│    Laravel      │
│   (Browser)     │    │ (Reverse Proxy) │    │   (Port 8000)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                └─────────────────┐
                                                  │
                               ┌─────────────────┐
                               │     Flask       │
                               │  AI Service     │
                               │   (Port 5000)   │
                               └─────────────────┘
```

### Route Distribution

- **Laravel** (`/`): Main application, web routes, frontend
- **Flask** (`/ai/*`): AI processing, machine learning endpoints
- **nginx**: Traffic routing, load balancing, SSL termination

## 📋 Prerequisites

### Windows Requirements

- **nginx for Windows** - [Download](https://nginx.org/en/download.html)
- **PHP 8.0+** - [Download](https://www.php.net/downloads.php)
- **Python 3.8+** - [Download](https://www.python.org/downloads/)
- **Composer** - [Download](https://getcomposer.org/download/)

### Verify Installations

```bash
php --version
python --version
composer --version
nginx -v
```

## 🚀 Quick Start

### 1. Clone and Setup Projects

```bash
# Clone your projects
git clone <your-laravel-repo>
git clone <your-flask-repo>

# Setup Laravel
cd laravel-project
composer install
cp .env.example .env
php artisan key:generate
php artisan migrate

# Setup Flask
cd ../flask-project
pip install -r requirements.txt
```

### 2. Configure nginx

Save the provided `nginx.conf` to your nginx installation directory:

```
C:\nginx\conf\nginx.conf
```

**For local development, update these lines:**

```nginx
server_name localhost;  # Change from deftection.com
```

### 3. Start All Services

**Terminal 1 - Laravel:**

```bash
cd laravel-project
php artisan serve --host=127.0.0.1 --port=8000
```

**Terminal 2 - Flask:**

```bash
cd flask-project
python app.py
# Flask should run on http://127.0.0.1:5000
```

**Terminal 3 - nginx (Run as Administrator):**

```cmd
cd C:\nginx
nginx.exe
```

## 🧪 Testing the Setup

### Health Checks

```bash
# Test Laravel (should show Laravel welcome page)
curl http://localhost/

# Test Flask AI endpoint
curl http://localhost/ai/health

# Check all services are running
netstat -ano | findstr :80
netstat -ano | findstr :8000
netstat -ano | findstr :5000
```

### Example API Calls

```javascript
// From your Laravel frontend to Flask AI
fetch("/ai/analyze", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    data: "your-data-here",
  }),
})
  .then((response) => response.json())
  .then((data) => console.log(data));
```

## ⚙️ Configuration

### Environment Variables

**Laravel (.env):**

```env
APP_URL=http://localhost
AI_SERVICE_URL=http://localhost/ai
```

**Flask:**

```python
# app.py
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for nginx integration

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
```

### nginx Configuration Features

- ✅ **Reverse Proxy**: Routes traffic between Laravel and Flask
- ✅ **CORS Enabled**: Browser can make direct API calls to Flask
- ✅ **Security Headers**: XSS protection, frame options, content sniffing
- ✅ **File Uploads**: 50MB maximum upload size
- ✅ **Error Handling**: Custom Laravel-powered error pages
- ✅ **Production Ready**: SSL/HTTPS configuration included (commented)

## 📁 Project Structure

```
project-root/
├── laravel-project/
│   ├── app/
│   ├── resources/
│   ├── routes/
│   └── .env
├── flask-project/
│   ├── app.py
│   ├── requirements.txt
│   └── models/
├── nginx.conf
└── README.md
```

## 🛠️ Development Scripts

### Batch Startup Script (Windows)

Create `start-servers.bat`:

```batch
@echo off
echo Starting Laravel...
start cmd /k "cd /d C:\path\to\laravel-project && php artisan serve --host=127.0.0.1 --port=8000"

echo Starting Flask...
start cmd /k "cd /d C:\path\to\flask-project && python app.py"

echo Starting nginx...
cd C:\nginx
nginx.exe

echo ✅ All servers started!
echo Laravel: http://localhost:8000
echo Flask: http://localhost:5000
echo Main App: http://localhost
pause
```

### Stop All Services

```batch
# stop-servers.bat
@echo off
echo Stopping nginx...
cd C:\nginx
nginx.exe -s stop

echo Stopping other services...
echo Press Ctrl+C in Laravel and Flask terminal windows
pause
```

## 🐛 Troubleshooting

### Common Issues

| Issue                   | Solution                                                          |
| ----------------------- | ----------------------------------------------------------------- |
| **502 Bad Gateway**     | Check Laravel/Flask are running on correct ports                  |
| **nginx won't start**   | Check `C:\nginx\logs\error.log` for details                       |
| **Port already in use** | Use `netstat -ano \| findstr :PORT` to find conflicting processes |
| **CORS errors**         | Ensure Flask has `CORS(app)` enabled                              |
| **404 on /ai routes**   | Verify Flask is accessible on port 5000                           |

### Port Conflicts

```bash
# Find what's using a port
netstat -ano | findstr :80

# Kill process (replace PID)
taskkill /PID <process_id> /F
```

### nginx Commands

```bash
# Test configuration
nginx -t

# Reload configuration
nginx -s reload

# Stop nginx
nginx -s stop

# View logs
type C:\nginx\logs\error.log
```

## 🚀 Production Deployment

### Enable HTTPS (Production)

1. **Obtain SSL Certificate:**

   ```bash
   # Using Let's Encrypt
   certbot --nginx -d yourdomain.com -d www.yourdomain.com
   ```

2. **Update nginx.conf:**

   - Uncomment the production SSL server blocks
   - Update domain names from `deftection.com` to your domain
   - Update SSL certificate paths

3. **Additional Security:**
   - Enable fail2ban for DDoS protection
   - Set up monitoring (Prometheus + Grafana)
   - Configure log rotation
   - Implement backup strategies

### Production Checklist

- [ ] SSL certificates configured
- [ ] Domain names updated
- [ ] Rate limiting enabled
- [ ] Security headers configured
- [ ] Monitoring setup
- [ ] Backup strategy implemented
- [ ] Firewall rules configured

## 📊 Monitoring

### Log Files

```bash
# nginx logs
tail -f C:\nginx\logs\access.log
tail -f C:\nginx\logs\error.log

# Laravel logs
tail -f laravel-project/storage/logs/laravel.log

# Flask logs (configure in your app.py)
```

### Health Endpoints

**Create health checks:**

```php
// Laravel - routes/web.php
Route::get('/health', function () {
    return response()->json(['status' => 'ok', 'service' => 'laravel']);
});
```

```python
# Flask - app.py
@app.route('/health')
def health():
    return {'status': 'ok', 'service': 'flask'}
```

**Made with ❤️ using Laravel + Flask + nginx**
