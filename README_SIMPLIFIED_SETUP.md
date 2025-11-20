# WealthArena Simplified Setup

This guide explains the simplified architecture for local development and testing.

## Architecture Overview

### What Changed?

- **Removed RAG System**: No vector database (Chroma) or embedding models
- **Direct GroQ API**: Chatbot uses GroQ API directly with guardrails
- **In-Memory Database**: Backend uses mock database for sessions
- **No Data Pipeline**: Uses mock data for testing

### Benefits

- Faster setup (no database installation)
- Fewer dependencies
- Easier testing
- Same functionality for development

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.9+
- Git
- GroQ API Key (get from https://console.groq.com)

### Run Setup

```powershell
.\master_setup_simplified.ps1
```

The script will:

1. Check prerequisites
2. Install dependencies
3. Configure environment
4. Start all services
5. Prompt you to test
6. Build APK

## Testing Checklist

When prompted, test these features:

### 1. Authentication

- [ ] Open http://localhost:8081
- [ ] Sign up with new account
- [ ] Log out
- [ ] Log in with same account

### 2. Onboarding

- [ ] Complete onboarding conversation
- [ ] Answer personality questions
- [ ] Select avatar
- [ ] Receive welcome rewards

### 3. Chatbot

- [ ] Ask financial questions (e.g., "What is RSI?")
- [ ] Try off-topic questions (should be rejected)
- [ ] Request trade setup for a symbol
- [ ] Verify educational disclaimers

### 4. App Features

- [ ] View dashboard
- [ ] Check portfolio
- [ ] Browse signals
- [ ] View leaderboard

## Guardrails

The chatbot is restricted to:

- Financial education topics
- WealthArena app features
- Trading concepts and risk management

Off-topic queries are politely rejected.

## APK Build

After testing, the script will:

1. Check EAS CLI login
2. Build Android APK
3. Provide download link

## Troubleshooting

### Services Won't Start

- Check if ports 3000, 8000, 8081 are available
- Verify GROQ_API_KEY is set
- Check Node.js and Python versions

### Chatbot Not Responding

- Verify GROQ_API_KEY in `chatbot/.env`
- Check chatbot logs in PowerShell window
- Test GroQ API key at https://console.groq.com

### Frontend Can't Connect

- Verify backend is running on port 3000
- Verify chatbot is running on port 8000
- Check CORS settings in backend

### Troubleshooting Physical Device Connectivity

Physical devices (Android phones, iPhones) need to connect to your development machine using its network IP address instead of `localhost`. The setup script automatically detects and configures this, but you may encounter issues.

#### Understanding the Problem

- **Emulators/Simulators**: Can use `localhost` or platform-specific URLs (e.g., `10.0.2.2` for Android emulator)
- **Physical Devices**: Must use your machine's actual IP address (e.g., `192.168.1.89`)

#### Finding Your Machine's IP Address

**Windows:**
```powershell
ipconfig
```
Look for "IPv4 Address" under your active network adapter (usually Wi-Fi or Ethernet).

**Mac/Linux:**
```bash
ifconfig
# or
ip addr
```
Look for the IP address on your active network interface (usually `en0` on Mac or `wlan0`/`eth0` on Linux).

#### Manual Configuration Steps

If the auto-detection fails or you need to update the IP address manually:

1. **Update Frontend Environment** (`frontend/.env.local`):
   ```env
   EXPO_PUBLIC_API_URL=http://YOUR_IP:3000
   EXPO_PUBLIC_BACKEND_URL=http://YOUR_IP:3000
   EXPO_PUBLIC_CHATBOT_URL=http://YOUR_IP:8000
   ```
   Replace `YOUR_IP` with your actual IP address (e.g., `192.168.1.89`).

2. **Update Backend CORS** (`backend/.env.local`):
   ```env
   CORS_ORIGINS=http://localhost:3000,http://localhost:8081,http://YOUR_IP:8081,http://YOUR_IP:3000
   ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8081,http://YOUR_IP:8081,http://YOUR_IP:3000
   ```
   Replace `YOUR_IP` with your actual IP address.

3. **Restart Services**: After updating the environment files, restart:
   - Backend service (port 3000)
   - Chatbot service (port 8000)
   - Frontend Expo server (port 8081)

#### Firewall Configuration

Windows Firewall may block incoming connections on ports 3000 and 8000. To allow connections:

**Option 1: Allow Node.js and Python through Firewall**
1. Open Windows Defender Firewall
2. Click "Allow an app or feature through Windows Defender Firewall"
3. Find "Node.js" and "Python" in the list
4. Check both "Private" and "Public" boxes for each
5. If not listed, click "Allow another app" and browse to:
   - Node.js: Usually at `C:\Program Files\nodejs\node.exe`
   - Python: Usually at `C:\Python3X\python.exe` or `C:\Users\YourName\AppData\Local\Programs\Python\Python3X\python.exe`

**Option 2: Create Inbound Rules for Ports**
1. Open Windows Defender Firewall
2. Click "Advanced settings"
3. Click "Inbound Rules" → "New Rule"
4. Select "Port" → Next
5. Select "TCP" and enter "3000" (repeat for port 8000)
6. Select "Allow the connection"
7. Apply to all profiles (Domain, Private, Public)
8. Name it "WealthArena Backend" (and "WealthArena Chatbot" for port 8000)

#### Network Requirements

- **Same Network**: Your physical device and development machine must be on the same Wi-Fi network
- **Network Restrictions**: Some corporate or public Wi-Fi networks block device-to-device communication. Try using a mobile hotspot or home network instead
- **VPN Interference**: VPNs may interfere with local network connectivity. Disable VPNs when testing on physical devices

#### Testing Connectivity

Before testing the app, verify connectivity:

1. **Test Backend Health Endpoint**:
   - On your physical device's browser, navigate to: `http://YOUR_IP:3000/health`
   - You should see a JSON response if the backend is accessible
   - If you get a connection error, check firewall settings and ensure the backend is running

2. **Test Chatbot Health Endpoint**:
   - Navigate to: `http://YOUR_IP:8000/health`
   - You should see a JSON response if the chatbot is accessible

3. **Check Network Info in App**:
   - The app logs network configuration information
   - Look for "Auto-detected backend IP" messages in the console
   - Verify the detected IP matches your machine's IP

#### IP Address Changes

If you change networks (e.g., switch from home Wi-Fi to mobile hotspot):

1. Find your new IP address using `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
2. Re-run the setup script: `.\master_setup_simplified.ps1` (it will detect the new IP)
3. Or manually update the environment files as described above
4. Restart all services

#### Common Issues

**"Network request failed" or "Connection refused"**
- Verify the IP address is correct in environment files
- Check that services are running on the correct ports
- Ensure firewall allows connections
- Verify device and computer are on the same network

**"CORS error" in browser console**
- Update `backend/.env.local` with your IP address in CORS_ORIGINS
- Restart the backend service after updating

**App connects but API calls fail**
- Check backend logs for errors
- Verify CORS configuration includes your IP
- Ensure backend is binding to `0.0.0.0` (not just `localhost`)

## Environment Variables

### Backend (.env.local)

**Note:** The setup script (`master_setup_simplified.ps1`) automatically generates this file with your machine's detected IP address during Phase 3. The file will be overwritten each time you run the setup script.

Example of what the script generates (with detected IP `192.168.1.89`):

```
# Simplified Local Development Configuration
USE_MOCK_DB=true
NODE_ENV=development
PORT=3000

# Mock Database (In-Memory) - no database credentials needed
# All user data will be stored in memory and cleared on server restart

# Service URLs
CHATBOT_URL=http://localhost:8000

# CORS Origins
# Includes localhost for emulators/simulators and machine IP for physical devices
CORS_ORIGINS=http://localhost:3000,http://localhost:8081,http://192.168.1.89:8081,http://192.168.1.89:3000
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8081,http://192.168.1.89:8081,http://192.168.1.89:3000

# JWT Secret
JWT_SECRET=local-dev-secret-change-in-production
```

If you need to manually configure this file, replace `192.168.1.89` with your actual machine IP address. See the [Troubleshooting Physical Device Connectivity](#troubleshooting-physical-device-connectivity) section for details.

### Chatbot (.env)

**⚠️ SECURITY WARNING:** Never commit your real `GROQ_API_KEY` to version control. The committed file is a template; the setup script generates a local version with your actual key. If your key is exposed, rotate it immediately at `https://console.groq.com`.

The setup script (`master_setup_simplified.ps1`) automatically generates this file during Phase 3, prompting you for your GroQ API key if not found.

```
# GroQ API Configuration
# Template: Replace with your actual key from https://console.groq.com. Do not commit real keys.
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192
LLM_PROVIDER=groq

# Server Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
```

### Frontend (.env.local)

**Note:** The setup script (`master_setup_simplified.ps1`) automatically generates this file with your machine's detected IP address during Phase 3. The file will be overwritten each time you run the setup script.

Example of what the script generates (with detected IP `192.168.1.89`):

```
# Backend API URL - uses machine IP for physical device connectivity
# Emulators/simulators will automatically use platform-specific localhost URLs
EXPO_PUBLIC_API_URL=http://192.168.1.89:3000
EXPO_PUBLIC_BACKEND_URL=http://192.168.1.89:3000

# Chatbot Service URL - uses machine IP for physical device connectivity
EXPO_PUBLIC_CHATBOT_URL=http://192.168.1.89:8000
```

If you need to manually configure this file, replace `192.168.1.89` with your actual machine IP address. See the [Troubleshooting Physical Device Connectivity](#troubleshooting-physical-device-connectivity) section for details.

