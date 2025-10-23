# WealthArena UI - Project Structure & Flow Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Directory Structure](#directory-structure)
4. [Application Flow](#application-flow)
5. [Component Architecture](#component-architecture)
6. [State Management](#state-management)
7. [Navigation Structure](#navigation-structure)
8. [Services & API Integration](#services--api-integration)
9. [Design System](#design-system)
10. [Development Workflow](#development-workflow)

---

## 🎯 Project Overview

**WealthArena UI** is a comprehensive React Native investment platform that gamifies financial learning and portfolio management. Built with Expo Router, it provides a complete ecosystem for investment simulation, portfolio management, and financial analytics.

### Key Technologies
- **React Native** with Expo Router
- **TypeScript** for type safety
- **NativeWind** for styling
- **Zustand** for state management
- **React Query** for server state
- **Lucide React Native** for icons

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    WealthArena UI                           │
├─────────────────────────────────────────────────────────────┤
│  📱 Mobile App (React Native + Expo)                       │
│  ├── 🎨 Design System (Custom Components)                  │
│  ├── 🧭 Navigation (Expo Router)                          │
│  ├── 🗃️ State Management (Zustand + React Query)          │
│  ├── 🔧 Services Layer (API Integration)                   │
│  └── 🎮 Gamification (VS AI Games, XP System)              │
├─────────────────────────────────────────────────────────────┤
│  🔗 Backend Integration                                     │
│  ├── 💬 Chatbot Service (Port 8000)                        │
│  ├── 🤖 RL Agent Service (Port 8001)                       │
│  └── 📊 Market Data APIs (Alpha Vantage, etc.)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
wealtharena_ui/
├── 📱 app/                          # Expo Router screens
│   ├── (tabs)/                      # Tab navigation screens
│   │   ├── dashboard.tsx            # Main dashboard
│   │   ├── opportunities.tsx       # Portfolio management
│   │   ├── game.tsx                # Investment games
│   │   ├── chat.tsx                # Leaderboard/chat
│   │   ├── account.tsx             # User profile
│   │   └── _layout.tsx             # Tab layout config
│   ├── ai-chat.tsx                  # AI chat interface
│   ├── trade-signals.tsx           # Trading signals
│   ├── trade-simulator.tsx         # Trading simulation
│   ├── portfolio-builder.tsx       # Portfolio creation
│   ├── strategy-lab.tsx            # Strategy testing
│   ├── analytics.tsx               # Analytics dashboard
│   ├── admin-portal.tsx            # Admin interface
│   ├── vs-ai-*.tsx                 # VS AI game screens
│   ├── landing.tsx                 # Landing page
│   ├── login.tsx                   # Authentication
│   ├── signup.tsx                  # User registration
│   ├── onboarding.tsx              # User onboarding
│   ├── splash.tsx                  # Splash screen
│   └── _layout.tsx                 # Root layout
├── 🎨 components/                   # Reusable components
│   ├── Charts/                      # Chart components
│   │   ├── ColumnChart.tsx
│   │   ├── PieChart.tsx
│   │   └── WorldMapView.tsx
│   ├── trade/                       # Trading components
│   │   ├── DurationSlider.tsx
│   │   ├── PlaybackControls.tsx
│   │   ├── ResultModal.tsx
│   │   ├── SimpleCandlestickChart.tsx
│   │   ├── StatusIndicator.tsx
│   │   ├── TradeActions.tsx
│   │   └── TradeLogPanel.tsx
│   ├── icons/                       # Custom icons
│   │   ├── FlatDogIcon.tsx
│   │   └── WealthArenaIcons.tsx
│   ├── AISignalCard.tsx            # AI signal display
│   ├── AnimatedGlow.tsx            # Animation effects
│   ├── AvatarSelector.tsx          # User avatar selection
│   ├── CandlestickChart.tsx        # Candlestick charts
│   ├── CharacterMascot.tsx         # Game mascot
│   ├── FloatingChatbot.tsx         # Chat interface
│   ├── GlassCard.tsx               # Glass morphism cards
│   ├── LeaderboardCard.tsx         # Leaderboard display
│   ├── TradeSetupCard.tsx          # Trade setup interface
│   └── UserAvatar.tsx              # User avatar
├── 🎨 src/design-system/           # Design system
│   ├── avatars/                    # Avatar components
│   │   ├── HumanAvatar.tsx
│   │   └── RobotAvatar.tsx
│   ├── mascots/                    # Game mascots
│   │   ├── FoxCelebrating.tsx
│   │   ├── FoxConfident.tsx
│   │   ├── FoxExcited.tsx
│   │   └── ... (10+ mascot states)
│   ├── icons/                      # Design system icons
│   │   ├── AgentIcon.tsx
│   │   ├── AlertIcon.tsx
│   │   ├── BellIcon.tsx
│   │   └── ... (20+ custom icons)
│   ├── Badge.tsx                   # Badge component
│   ├── Button.tsx                  # Button component
│   ├── Card.tsx                    # Card component
│   ├── FAB.tsx                     # Floating Action Button
│   ├── Header.tsx                  # Header component
│   ├── ProgressRing.tsx            # Progress indicators
│   ├── Sparkline.tsx               # Mini charts
│   ├── Text.tsx                    # Typography
│   ├── TextInput.tsx               # Input components
│   ├── ThemeProvider.tsx           # Theme management
│   └── tokens.ts                   # Design tokens
├── 🗃️ contexts/                     # React contexts
│   ├── SimulationContext.tsx       # Trading simulation state
│   ├── ThemeContext.tsx            # Theme management
│   ├── UserContext.tsx             # User state
│   ├── UserSettingsContext.tsx     # User preferences
│   └── UserTierContext.tsx         # User tier system
├── 🔧 services/                    # API services
│   ├── aiSignalAdapter.ts          # AI signal processing
│   ├── alphaVantageService.ts      # Market data API
│   ├── apiService.ts               # Base API service
│   ├── assetService.ts             # Asset management
│   ├── chatbotService.ts           # Chatbot integration
│   ├── marketDataService.ts        # Market data
│   ├── newsService.ts              # News integration
│   ├── portfolioService.ts         # Portfolio management
│   ├── recommendationService.ts    # AI recommendations
│   └── rlAgentService.ts           # RL agent integration
├── 📊 data/                         # Mock data
│   ├── historicalData.ts           # Historical market data
│   ├── leaderboardData.ts          # Leaderboard data
│   └── mockCandleData.ts           # Candlestick data
├── 🗄️ database/                     # Database setup
│   ├── AzureSQL_CreateTables.sql   # Database schema
│   ├── chatbot-helper.ts           # Chatbot utilities
│   ├── db-connection.ts            # Database connection
│   └── test-connection.ts          # Connection testing
├── 🎯 types/                        # TypeScript types
│   ├── ai-signal.ts                # AI signal types
│   ├── candlestick.ts              # Chart types
│   └── global.d.ts                 # Global type definitions
├── 🛠️ utils/                        # Utility functions
│   ├── aiTrader.ts                 # AI trading logic
│   ├── linking.ts                  # Deep linking
│   └── simulationEngine.ts         # Trading simulation
├── 📋 constants/                    # App constants
│   └── colors.ts                   # Color definitions
├── ⚙️ config/                       # Configuration
│   └── apiKeys.example.ts          # API key templates
├── 📄 Documentation Files
│   ├── README.md                   # Main documentation
│   ├── PROJECT_STRUCTURE_AND_FLOW.md # This file
│   ├── FIXED_INTEGRATION_SUMMARY.md # Integration status
│   ├── BACKEND_INTEGRATION_GUIDE.md # Backend setup
│   ├── DESIGN_SYSTEM_GUIDE.md      # Design system docs
│   ├── TESTING_GUIDE.md            # Testing instructions
│   └── ... (20+ documentation files)
├── 📦 Configuration Files
│   ├── package.json                # Dependencies
│   ├── app.json                    # Expo configuration
│   ├── tsconfig.json              # TypeScript config
│   ├── babel.config.js             # Babel configuration
│   ├── metro.config.js             # Metro bundler config
│   ├── eslint.config.js           # ESLint configuration
│   └── polyfills.js                # Polyfills
└── 🎨 Assets
    └── dog.svg                     # App icon
```

---

## 🔄 Application Flow

### 1. **App Initialization Flow**
```
Splash Screen → Landing Page → Authentication → Onboarding → Main App
```

### 2. **Authentication Flow**
```
Landing → Login/Signup → Onboarding → Dashboard
```

### 3. **Main Navigation Flow**
```
Tab Navigation:
├── Dashboard (Home)
├── Portfolio (Opportunities)
├── Game (VS AI)
├── Leaderboard (Chat)
└── Account (Profile)
```

### 4. **Trading Flow**
```
Dashboard → Trade Signals → Trade Setup → Trade Simulator → Results
```

### 5. **Game Flow**
```
Dashboard → VS AI Start → VS AI Battle → VS AI Play → VS AI Game Over
```

### 6. **Portfolio Management Flow**
```
Dashboard → Portfolio Builder → Strategy Lab → Analytics
```

---

## 🧩 Component Architecture

### **Design System Hierarchy**
```
ThemeProvider
├── Design Tokens (colors, typography, spacing)
├── Base Components (Button, Card, Text, Input)
├── Composite Components (FAB, Header, ProgressRing)
├── Specialized Components (Charts, Mascots, Icons)
└── Screen Components (Dashboard, Game, Profile)
```

### **Component Categories**

#### **1. Base Components**
- `Button.tsx` - Interactive buttons
- `Card.tsx` - Container components
- `Text.tsx` - Typography system
- `TextInput.tsx` - Form inputs
- `Badge.tsx` - Status indicators

#### **2. Specialized Components**
- `CandlestickChart.tsx` - Financial charts
- `FloatingChatbot.tsx` - AI chat interface
- `CharacterMascot.tsx` - Game mascot
- `AISignalCard.tsx` - AI signal display
- `LeaderboardCard.tsx` - Leaderboard entries

#### **3. Trading Components**
- `DurationSlider.tsx` - Time selection
- `PlaybackControls.tsx` - Simulation controls
- `TradeActions.tsx` - Trade execution
- `ResultModal.tsx` - Trade results
- `StatusIndicator.tsx` - Trade status

#### **4. Chart Components**
- `ColumnChart.tsx` - Bar charts
- `PieChart.tsx` - Pie charts
- `WorldMapView.tsx` - Geographic data
- `Sparkline.tsx` - Mini trend charts

---

## 🗃️ State Management

### **Context Providers**
```typescript
QueryClientProvider
├── ThemeProvider
│   ├── UserProvider
│   │   ├── UserTierProvider
│   │   │   └── UserSettingsProvider
│   │   │       └── App Components
```

### **State Categories**

#### **1. User State (UserContext)**
- User profile information
- Authentication status
- User preferences
- Session management

#### **2. User Tier State (UserTierContext)**
- Bronze, Silver, Gold, Platinum tiers
- Tier-based feature access
- XP and achievement tracking
- Progress indicators

#### **3. User Settings (UserSettingsContext)**
- Theme preferences (dark/light)
- Notification settings
- Privacy preferences
- App configuration

#### **4. Simulation State (SimulationContext)**
- Trading simulation data
- Portfolio performance
- Market conditions
- Trade history

#### **5. Theme State (ThemeContext)**
- Color schemes
- Typography settings
- Component styling
- Dark/light mode

---

## 🧭 Navigation Structure

### **Root Navigation (Stack)**
```
RootLayout
├── index (Landing)
├── splash
├── landing
├── login
├── signup
├── onboarding
├── (tabs) - Main App
├── ai-chat
├── trade-signals
├── trade-simulator
├── portfolio-builder
├── strategy-lab
├── analytics
├── admin-portal
├── vs-ai-start
├── vs-ai-battle
├── vs-ai-play
├── vs-ai-gameover
└── +not-found
```

### **Tab Navigation**
```
(tabs)
├── dashboard (Home)
├── opportunities (Portfolio)
├── game (VS AI Games)
├── chat (Leaderboard)
└── account (Profile)
```

### **Navigation Features**
- **Stack Navigation**: Screen transitions
- **Tab Navigation**: Bottom tab bar
- **Modal Presentation**: Overlay screens
- **Deep Linking**: URL-based navigation
- **Gesture Navigation**: Swipe gestures

---

## 🔧 Services & API Integration

### **Service Architecture**
```
Frontend Services
├── apiService.ts (Base API client)
├── chatbotService.ts (AI Chatbot)
├── rlAgentService.ts (RL Agent)
├── marketDataService.ts (Market data)
├── portfolioService.ts (Portfolio)
├── newsService.ts (News feed)
└── recommendationService.ts (AI recommendations)
```

### **External Integrations**
```
Backend Services
├── Chatbot API (Port 8000)
│   ├── Natural language processing
│   ├── Investment advice
│   └── Market insights
├── RL Agent API (Port 8001)
│   ├── Trading signals
│   ├── Strategy recommendations
│   └── Risk assessment
└── Market Data APIs
    ├── Alpha Vantage
    ├── Real-time prices
    └── Historical data
```

### **API Service Pattern**
```typescript
// Service structure
class ApiService {
  private baseURL: string;
  private headers: Record<string, string>;
  
  async get<T>(endpoint: string): Promise<T>
  async post<T>(endpoint: string, data: any): Promise<T>
  async put<T>(endpoint: string, data: any): Promise<T>
  async delete<T>(endpoint: string): Promise<T>
}
```

---

## 🎨 Design System

### **Design Tokens**
```typescript
// tokens.ts
export const tokens = {
  color: {
    primary: '#007AFF',
    secondary: '#5856D6',
    accentYellow: '#FFD60A',
    // ... color palette
  },
  font: {
    sizes: { xs: 12, sm: 14, base: 16, lg: 18, xl: 20 },
    weights: { normal: '400', medium: '500', semibold: '600', bold: '700' }
  },
  spacing: { xs: 4, sm: 8, md: 16, lg: 24, xl: 32 },
  borderRadius: { sm: 4, md: 8, lg: 12, xl: 16 }
}
```

### **Theme System**
```typescript
// Theme structure
interface Theme {
  bg: string;           // Background color
  surface: string;      // Surface color
  primary: string;      // Primary color
  secondary: string;    // Secondary color
  text: string;         // Text color
  muted: string;        // Muted text
  border: string;       // Border color
  success: string;      // Success color
  warning: string;      // Warning color
  error: string;        // Error color
}
```

### **Component Variants**
- **Buttons**: Primary, Secondary, Ghost, Danger
- **Cards**: Default, Glass, Elevated, Outlined
- **Text**: Heading, Body, Caption, Label
- **Inputs**: Default, Filled, Outlined, Underlined

---

## 🚀 Development Workflow

### **Project Setup**
```bash
# Install dependencies
bun install

# Start development server
bun run start          # Mobile development
bun run start-web       # Web development
bun run start-web-dev   # Web with debug logs
```

### **Development Commands**
```bash
# Linting
bun run lint

# Type checking
npx tsc --noEmit

# Build for production
eas build --platform ios
eas build --platform android
eas build --platform web
```

### **File Organization Rules**
1. **Screens**: Place in `app/` directory
2. **Components**: Place in `components/` directory
3. **Services**: Place in `services/` directory
4. **Types**: Place in `types/` directory
5. **Utils**: Place in `utils/` directory
6. **Contexts**: Place in `contexts/` directory

### **Code Style Guidelines**
- Use TypeScript for all files
- Follow React Native best practices
- Use functional components with hooks
- Implement proper error handling
- Add JSDoc comments for complex functions
- Use consistent naming conventions

---

## 📱 Platform Support

### **Mobile Platforms**
- **iOS**: Native iOS app with iOS-specific optimizations
- **Android**: Native Android app with Material Design
- **Web**: Progressive Web App (PWA) support

### **Development Tools**
- **Expo CLI**: Development and building
- **EAS Build**: Cloud building service
- **EAS Submit**: App store submission
- **Expo Go**: Development testing

### **Deployment Pipeline**
```
Development → Testing → Staging → Production
     ↓           ↓         ↓          ↓
  Local Dev   Expo Go   EAS Build   App Stores
```

---

## 🔗 Integration Points

### **Backend Services**
- **Chatbot Service**: AI-powered investment advice
- **RL Agent Service**: Machine learning trading signals
- **Market Data APIs**: Real-time financial data
- **User Management**: Authentication and profiles
- **Portfolio Management**: Investment tracking

### **External APIs**
- **Alpha Vantage**: Market data and stock prices
- **News APIs**: Financial news and updates
- **Analytics**: User behavior tracking
- **Push Notifications**: Real-time alerts

---

## 📊 Performance Considerations

### **Optimization Strategies**
- **Lazy Loading**: Screen-based code splitting
- **Image Optimization**: Efficient asset loading
- **State Management**: Minimal re-renders
- **API Caching**: React Query for data caching
- **Bundle Size**: Tree shaking and code splitting

### **Monitoring**
- **Performance Metrics**: App speed and responsiveness
- **Error Tracking**: Crash reporting and debugging
- **User Analytics**: Usage patterns and engagement
- **API Monitoring**: Service health and response times

---

## 🧪 Testing Strategy

### **Testing Levels**
- **Unit Tests**: Component and utility testing
- **Integration Tests**: Service and API testing
- **E2E Tests**: Complete user flow testing
- **Performance Tests**: Load and stress testing

### **Testing Tools**
- **Jest**: Unit testing framework
- **React Native Testing Library**: Component testing
- **Detox**: End-to-end testing
- **Flipper**: Debugging and inspection

---

## 📚 Documentation

### **Documentation Files**
- `README.md` - Main project documentation
- `PROJECT_STRUCTURE_AND_FLOW.md` - This file
- `DESIGN_SYSTEM_GUIDE.md` - Design system documentation
- `BACKEND_INTEGRATION_GUIDE.md` - Backend setup guide
- `TESTING_GUIDE.md` - Testing instructions
- `DEPLOYMENT_CHECKLIST.md` - Deployment guide

### **Code Documentation**
- JSDoc comments for functions
- TypeScript interfaces for data structures
- README files in component directories
- Inline comments for complex logic

---

## 🎯 Future Enhancements

### **Planned Features**
- **Advanced Analytics**: More detailed performance metrics
- **Social Features**: User interactions and sharing
- **Advanced AI**: More sophisticated AI recommendations
- **Offline Support**: Offline trading simulation
- **Multi-language**: Internationalization support

### **Technical Improvements**
- **Performance**: Further optimization and caching
- **Accessibility**: Enhanced accessibility features
- **Security**: Enhanced security measures
- **Scalability**: Better handling of large datasets

---

This documentation provides a comprehensive overview of the WealthArena UI project structure and flow. It serves as a reference for developers, designers, and stakeholders to understand the application architecture, component organization, and development workflow.
