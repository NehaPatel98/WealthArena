# WealthArena Mobile Integration

🚀 **Mobile SDKs and UI Components** for WealthArena Trading Education Platform

Transform your existing trading-education + chatbot logic into a mobile-integration block that plugs into existing apps without altering their codebase.

## 📱 What's Included

### 1. Backend API (FastAPI)
- **Versioned endpoints**: `/v1/chat`, `/v1/analyze`, `/v1/state`, `/v1/papertrade`, `/v1/learn`, `/healthz`
- **Optional authentication**: Bearer token support
- **CORS configured**: For mobile emulators and localhost
- **OpenAPI spec**: Auto-generated at `/openapi.json`

### 2. Mobile SDKs
- **React Native (TypeScript)**: `@wealtharena/mobile-sdk-rn`
- **Android (Kotlin)**: Retrofit + OkHttp + Coroutines
- **iOS (Swift)**: URLSession + async/await

### 3. Optional UI Components
- **React Native**: `@wealtharena/wealtharena-rn` with black/neon-green theme
- **Drop-in component**: `<WealthArenaScreen />` with Learn | Analyze | Trade | Chat tabs

## 🚀 Quick Start

### 1. Start the Backend

```bash
# Install dependencies
uv venv && source .venv/bin/activate
uv add fastapi uvicorn black ruff mypy pytest pydantic-settings python-dotenv

# Start the server
uvicorn src.bot.app:app --reload --host 0.0.0.0 --port 8000
```

### 2. Test the API

```bash
# Test mobile API
python test_mobile_api.py

# Manual test
curl http://localhost:8000/v1/healthz
```

### 3. Use Mobile SDKs

#### React Native
```typescript
import { createWealthArenaClient } from '@wealtharena/mobile-sdk-rn';
import { WealthArenaScreen } from '@wealtharena/wealtharena-rn';

const client = createWealthArenaClient('http://127.0.0.1:8000');

// Use the drop-in UI component
<WealthArenaScreen 
  client={client}
  onEvent={(event) => console.log('Event:', event)}
/>
```

#### Android
```kotlin
import com.wealtharena.mobile.sdk.*

val client = createWealthArenaClient("http://10.0.2.2:8000")

lifecycleScope.launch {
    val response = client.chat(ChatRequest("Explain RSI"))
    // Handle response
}
```

#### iOS
```swift
import WealthArenaSDK

let client = createWealthArenaClient(baseURL: URL(string: "http://127.0.0.1:8000")!)

Task {
    let response = try await client.chat(ChatRequest(message: "Explain RSI"))
    // Handle response
}
```

## 📁 Project Structure

```
WealthArena/
├── src/bot/
│   ├── app.py              # Main FastAPI app
│   ├── api_v1.py           # Versioned mobile API
│   └── kb.py               # Knowledge base
├── packages/
│   ├── mobile-sdk-rn/      # React Native SDK
│   ├── wealtharena-rn/     # React Native UI components
│   ├── mobile-sdk-android/ # Android SDK
│   └── mobile-sdk-ios/     # iOS SDK
├── examples/
│   ├── rn-demo/            # React Native demo app
│   ├── android-demo/       # Android demo app
│   └── ios-demo/           # iOS demo app
├── docs/
│   ├── INTEGRATION_RN.md   # React Native integration guide
│   ├── INTEGRATION_ANDROID.md # Android integration guide
│   └── INTEGRATION_IOS.md # iOS integration guide
└── scripts/
    └── build-sdks.sh       # Build all SDKs
```

## 🔧 Configuration

### Environment Variables

Create `.env` file:

```env
BASE_URL=http://127.0.0.1:8000
AUTH_REQUIRED=false
API_TOKEN=wealtharena-mobile-token
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://10.0.2.2:8000
```

### Mobile Emulator URLs

- **Android Emulator**: `http://10.0.2.2:8000`
- **iOS Simulator**: `http://127.0.0.1:8000`
- **React Native**: `http://localhost:8000`

## 📚 API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/healthz` | GET | Health check |
| `/v1/chat` | POST | Chat with trading bot |
| `/v1/analyze` | POST | Analyze asset with technical indicators |
| `/v1/state` | GET | Get trading state (balance, positions) |
| `/v1/papertrade` | POST | Execute paper trade |
| `/v1/learn` | GET | Get educational content |

## 🛠️ Development

### Build All SDKs

```bash
# Make script executable
chmod +x scripts/build-sdks.sh

# Build all SDKs
./scripts/build-sdks.sh
```

### Run Tests

```bash
# Backend tests
python test_mobile_api.py

# React Native SDK tests
cd packages/mobile-sdk-rn && npm test

# Android SDK tests
cd packages/mobile-sdk-android && ./gradlew test

# iOS SDK tests
cd packages/mobile-sdk-ios && swift test
```

### Run Demo Apps

```bash
# React Native demo
cd examples/rn-demo
npm install
npm start

# Android demo
cd examples/android-demo
./gradlew installDebug

# iOS demo
cd examples/ios-demo
open WealthArenaDemo.xcodeproj
```

## 🎨 UI Components

### React Native Theme

The React Native UI components use a black/neon-green terminal theme:

```typescript
const customTheme = {
  colors: {
    background: '#0b0f12',    // Dark background
    primary: '#00ff88',       // Neon green
    surface: '#1a1f24',       // Card background
    text: '#ffffff',          // White text
    textSecondary: '#a0a0a0'  // Gray text
  }
};
```

### Event Handling

The UI components emit events for analytics:

```typescript
<WealthArenaScreen 
  client={client}
  onEvent={(event) => {
    switch (event.type) {
      case 'ChatMessage':
        analytics.track('chat_message', event.data);
        break;
      case 'TradePlaced':
        analytics.track('paper_trade', event.data);
        break;
      case 'AnalysisGenerated':
        analytics.track('analysis_request', event.data);
        break;
    }
  }}
/>
```

## 🔒 Security

### Authentication

Optional bearer token authentication:

```bash
# Set AUTH_REQUIRED=true in .env
curl -H "Authorization: Bearer your-token" \
  http://localhost:8000/v1/state
```

### CORS Configuration

Configured for mobile development:

- `http://localhost:3000` - React Native Metro
- `http://127.0.0.1:3000` - Local development
- `http://10.0.2.2:8000` - Android emulator
- `http://localhost:19006` - Expo

## 🚀 Deployment

### Production Backend

```bash
# Set production environment
export BASE_URL=https://api.wealtharena.com
export AUTH_REQUIRED=true
export API_TOKEN=your-production-token

# Start with production settings
uvicorn src.bot.app:app --host 0.0.0.0 --port 8000
```

### Mobile App Integration

1. **Update base URL** in your mobile app
2. **Add authentication token** if required
3. **Test with production backend**
4. **Deploy to app stores**

## 🐛 Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure backend is running on correct port
   - Check firewall settings
   - Use correct emulator URLs

2. **CORS Errors**
   - Verify CORS origins in backend
   - Check mobile emulator configuration

3. **Authentication Errors**
   - Verify token format: `Bearer your-token`
   - Check `AUTH_REQUIRED` setting

4. **Build Errors**
   - Ensure all dependencies are installed
   - Check TypeScript/Swift/Kotlin versions
   - Clear build caches

### Debug Mode

Enable debug logging:

```typescript
// React Native
const client = createWealthArenaClient('http://127.0.0.1:8000', {
  debug: true
});
```

## 📞 Support

- **Documentation**: [Integration Guides](docs/)
- **Issues**: [GitHub Issues](https://github.com/wealtharena/mobile-sdk/issues)
- **Email**: support@wealtharena.com

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**⚠️ Educational Only**: This platform is for educational purposes only. Not financial advice. Always practice with paper trading first.