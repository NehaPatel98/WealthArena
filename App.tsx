import React, { useState, useEffect } from 'react';
import {
  StatusBar,
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  useColorScheme,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { colors } from './src/theme/colors';

// Import screens
import DashboardScreen from './src/screens/DashboardScreen';
import AlertsScreen from './src/screens/AlertsScreen';
import ChallengesScreen from './src/screens/ChallengesScreen';
import ProfileScreen from './src/screens/ProfileScreen';
import StrategyLabScreen from './src/screens/StrategyLabScreen';
import LoginScreen from './src/screens/LoginScreen';
import SignupScreen from './src/screens/SignupScreen';
import OnboardingScreen from './src/screens/OnboardingScreen';
import LandingScreen from './src/screens/LandingScreen';
import PortfolioBuilderScreen from './src/screens/PortfolioBuilderScreen';
import GameModeScreen from './src/screens/GameModeScreen';
import TradeSimulatorScreen from './src/screens/TradeSimulatorScreen';
import AnalyticsScreen from './src/screens/AnalyticsScreen';
import NotificationsScreen from './src/screens/NotificationsScreen';

interface UserProfile {
  experience: 'beginner' | 'intermediate' | 'advanced';
  riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  investmentGoals: string[];
  timeHorizon: 'short' | 'medium' | 'long';
  suggestedPortfolio: string[];
}

type AppState = 'landing' | 'onboarding' | 'login' | 'signup' | 'authenticated' | 'portfolio-builder' | 'strategy-lab' | 'game-mode' | 'trade-simulator' | 'analytics' | 'notifications';

const App = () => {
  const [appState, setAppState] = useState<AppState>('landing');
  const [activeTab, setActiveTab] = useState('Dashboard');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const colorScheme = useColorScheme();
  const isDarkMode = true; // Force dark mode
  const c = colors[isDarkMode ? 'dark' : 'light'];

  useEffect(() => {
    checkAuthStatus();
  }, []);

  const checkAuthStatus = async () => {
    try {
      const token = await AsyncStorage.getItem('authToken');
      const hasSeenOnboarding = await AsyncStorage.getItem('hasSeenOnboarding');
      const savedProfile = await AsyncStorage.getItem('userProfile');
      
      if (token) {
        setIsAuthenticated(true);
        setAppState('authenticated');
        if (savedProfile) {
          setUserProfile(JSON.parse(savedProfile));
        }
      } else if (!hasSeenOnboarding) {
        setAppState('landing');
      } else {
        setAppState('login');
      }
    } catch (error) {
      console.error('Error checking auth status:', error);
      setAppState('landing');
    }
  };

  const handleLogin = async (email: string, password: string) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Store auth token
    await AsyncStorage.setItem('authToken', 'mock-token');
    setIsAuthenticated(true);
    setAppState('authenticated');
  };

  const handleSignup = async (fullName: string, email: string, password: string, confirmPassword: string) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Store auth token
    await AsyncStorage.setItem('authToken', 'mock-token');
    setIsAuthenticated(true);
    setAppState('authenticated');
  };

  const handleOnboardingComplete = async (profile?: UserProfile) => {
    await AsyncStorage.setItem('hasSeenOnboarding', 'true');
    if (profile) {
      setUserProfile(profile);
      await AsyncStorage.setItem('userProfile', JSON.stringify(profile));
    }
    setAppState('login');
  };

  const handleLogout = async () => {
    await AsyncStorage.removeItem('authToken');
    setIsAuthenticated(false);
    setAppState('login');
  };

  // Render different screens based on app state
  if (appState === 'landing') {
    return (
      <LandingScreen
        onNavigateToAuth={() => setAppState('login')}
        onNavigateToOnboarding={() => setAppState('onboarding')}
      />
    );
  }

  if (appState === 'onboarding') {
    return <OnboardingScreen onComplete={handleOnboardingComplete} />;
  }

  if (appState === 'login') {
    return (
      <LoginScreen
        onLogin={handleLogin}
        onNavigateToSignup={() => setAppState('signup')}
      />
    );
  }

  if (appState === 'signup') {
    return (
      <SignupScreen
        onSignup={handleSignup}
        onNavigateToLogin={() => setAppState('login')}
      />
    );
  }

  if (appState === 'portfolio-builder' && userProfile) {
    return (
      <PortfolioBuilderScreen
        userProfile={userProfile}
        onPortfolioCreate={(portfolio) => {
          console.log('Portfolio created:', portfolio);
          setAppState('authenticated');
        }}
        onBack={() => setAppState('authenticated')}
      />
    );
  }

  if (appState === 'strategy-lab' && userProfile) {
    return (
      <StrategyLabScreen
        userProfile={userProfile}
        onBack={() => setAppState('authenticated')}
      />
    );
  }

  if (appState === 'game-mode' && userProfile) {
    return (
      <GameModeScreen
        userProfile={userProfile}
        onBack={() => setAppState('authenticated')}
      />
    );
  }

  if (appState === 'trade-simulator' && userProfile) {
    return (
      <TradeSimulatorScreen
        userProfile={userProfile}
        onBack={() => setAppState('authenticated')}
      />
    );
  }

  if (appState === 'analytics' && userProfile) {
    return (
      <AnalyticsScreen
        userProfile={userProfile}
        onBack={() => setAppState('authenticated')}
      />
    );
  }

  if (appState === 'notifications') {
    return (
      <NotificationsScreen
        onBack={() => setAppState('authenticated')}
      />
    );
  }

  // Only show the main app interface when authenticated
  if (appState !== 'authenticated') {
    return null; // or a loading screen
  }

  const backgroundStyle = {
    backgroundColor: c.background,
    flex: 1,
  };

  const renderScreen = () => {
    switch (activeTab) {
      case 'Dashboard':
        return (
          <DashboardScreen 
            onNavigateToPortfolioBuilder={() => setAppState('portfolio-builder')}
            onNavigateToStrategyLab={() => setAppState('strategy-lab')}
            onNavigateToGameMode={() => setAppState('game-mode')}
            onNavigateToTradeSimulator={() => setAppState('trade-simulator')}
            onNavigateToAnalytics={() => setAppState('analytics')}
            onNavigateToNotifications={() => setAppState('notifications')}
          />
        );
      case 'Alerts':
        return <AlertsScreen />;
      case 'Challenges':
        return <ChallengesScreen />;
      case 'Strategy':
        return <StrategyLabScreen userProfile={userProfile!} onBack={() => setActiveTab('Dashboard')} />;
      case 'Profile':
        return <ProfileScreen onLogout={handleLogout} />;
      default:
        return (
          <DashboardScreen 
            onNavigateToPortfolioBuilder={() => setAppState('portfolio-builder')}
            onNavigateToStrategyLab={() => setAppState('strategy-lab')}
            onNavigateToGameMode={() => setAppState('game-mode')}
            onNavigateToTradeSimulator={() => setAppState('trade-simulator')}
            onNavigateToAnalytics={() => setAppState('analytics')}
            onNavigateToNotifications={() => setAppState('notifications')}
          />
        );
    }
  };

  const tabs = [
    { id: 'Dashboard', icon: 'view-dashboard', label: 'Dashboard' },
    { id: 'Alerts', icon: 'bell', label: 'Alerts' },
    { id: 'Challenges', icon: 'trophy', label: 'Challenges' },
    { id: 'Strategy', icon: 'flask', label: 'Strategy' },
    { id: 'Profile', icon: 'account', label: 'Profile' },
  ];

  return (
        <View style={backgroundStyle}>
          <StatusBar
            barStyle={isDarkMode ? "light-content" : "dark-content"}
            backgroundColor={c.background}
          />
      
      {/* Main Content */}
      <View style={styles.content}>
        {renderScreen()}
      </View>

          {/* Bottom Tab Navigation */}
          <View style={[
            styles.tabBar,
            {
              backgroundColor: c.surface,
              borderTopColor: c.border,
              shadowColor: c.shadow.medium,
              shadowOffset: { width: 0, height: -2 },
              shadowOpacity: 1,
              shadowRadius: 8,
              elevation: 8,
            }
          ]}>
            {tabs.map((tab) => (
              <TouchableOpacity
                key={tab.id}
                style={[
                  styles.tab,
                  activeTab === tab.id && {
                    backgroundColor: c.primary + '20',
                    borderRadius: 12,
                    marginHorizontal: 4,
                  }
                ]}
                onPress={() => setActiveTab(tab.id)}
              >
                <Icon
                  name={activeTab === tab.id ? tab.icon : `${tab.icon}-outline`}
                  size={24}
                  color={activeTab === tab.id ? c.primary : c.textMuted}
                />
                <Text style={[
                  styles.tabLabel,
                  {
                    color: activeTab === tab.id ? c.primary : c.textMuted,
                    fontWeight: activeTab === tab.id ? '700' : '500',
                  }
                ]}>
                  {tab.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
    </View>
  );
};

const styles = StyleSheet.create({
  content: {
    flex: 1,
  },
  tabBar: {
    flexDirection: 'row',
    borderTopWidth: 1,
    paddingVertical: 12,
    paddingHorizontal: 4,
    minHeight: 60,
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: -2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3,
  },
  tab: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 8,
    justifyContent: 'center',
  },
  tabLabel: {
    fontSize: 11,
    marginTop: 4,
    fontWeight: '600',
  },
});

export default App;