import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  Dimensions,
  useColorScheme,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { colors } from '../theme/colors';

interface LandingScreenProps {
  onNavigateToAuth: () => void;
  onNavigateToOnboarding: () => void;
}

const { width } = Dimensions.get('window');

const LandingScreen: React.FC<LandingScreenProps> = ({ onNavigateToAuth, onNavigateToOnboarding }) => {
  const [currentFeature, setCurrentFeature] = useState(0);
  const colorScheme = useColorScheme();
  const isDarkMode = true;
  const c = colors[isDarkMode ? 'dark' : 'light'];

  const features = [
    {
      icon: 'chart-line',
      title: 'Real-Time Analytics',
      description: 'Advanced portfolio analytics with P&L tracking, risk metrics, and factor attribution',
      color: c.primary,
    },
    {
      icon: 'robot',
      title: 'AI-Powered Insights',
      description: 'Dynamic AI chatbot that adapts to your skill level and provides personalized recommendations',
      color: c.success,
    },
    {
      icon: 'gamepad-variant',
      title: 'Historical Game Mode',
      description: 'Learn by playing through historical market scenarios with real-time decision making',
      color: c.warning,
    },
    {
      icon: 'flask',
      title: 'Strategy Lab',
      description: 'Build, test, and optimize trading strategies with advanced backtesting tools',
      color: c.accent,
    },
    {
      icon: 'shield-check',
      title: 'Secure & Private',
      description: 'Bank-level security with end-to-end encryption and SOC 2 compliance',
      color: '#4CAF50',
    },
    {
      icon: 'trophy',
      title: 'Competition & Learning',
      description: 'Compete with AI agents and other users in tournaments and challenges',
      color: '#FF9800',
    },
  ];

  const screenshots = [
    { title: 'Dashboard Overview', description: 'Portfolio snapshot with P&L, allocations, and risk metrics' },
    { title: 'AI Chatbot Onboarding', description: 'Dynamic skill assessment and tier categorization' },
    { title: 'Historical Game Mode', description: 'Play through 2008 crash, dot-com bubble, and more' },
    { title: 'Strategy Lab', description: 'Build and backtest trading strategies with RL agents' },
    { title: 'Portfolio Builder', description: 'Create portfolios with constraints and stress testing' },
    { title: 'Analytics Dashboard', description: 'Advanced performance metrics and factor attribution' },
    { title: 'Competition Leaderboard', description: 'Compete with AI agents and other users' },
    { title: 'Trade Simulator', description: 'Practice trading with realistic market conditions' },
  ];

  return (
    <ScrollView style={[styles.container, { backgroundColor: c.background }]} showsVerticalScrollIndicator={false}>
      {/* Header */}
      <View style={styles.header}>
        <View style={[styles.logoContainer, { backgroundColor: c.primary }]}>
          <Icon name="trending-up" size={32} color={c.background} />
        </View>
        <Text style={[styles.logoText, { color: c.text }]}>WealthArena</Text>
        <View style={styles.headerButtons}>
          <TouchableOpacity style={[styles.loginButton, { borderColor: c.primary }]} onPress={onNavigateToAuth}>
            <Text style={[styles.loginButtonText, { color: c.primary }]}>Sign In</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Hero Section */}
      <View style={styles.hero}>
        <Text style={[styles.heroTitle, { color: c.text }]}>
          Master Your Financial Future
        </Text>
        <Text style={[styles.heroSubtitle, { color: c.textMuted }]}>
          AI-powered investment platform that adapts to your skill level and helps you build wealth intelligently
        </Text>
        <View style={styles.heroButtons}>
          <TouchableOpacity 
            style={[styles.ctaButton, { backgroundColor: c.primary }]} 
            onPress={onNavigateToOnboarding}
          >
            <Text style={[styles.ctaButtonText, { color: c.background }]}>Start Free Trial</Text>
            <Icon name="arrow-right" size={20} color={c.background} />
          </TouchableOpacity>
          <TouchableOpacity style={[styles.secondaryButton, { borderColor: c.border }]} onPress={onNavigateToAuth}>
            <Text style={[styles.secondaryButtonText, { color: c.text }]}>Learn More</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Features Section */}
      <View style={styles.featuresSection}>
        <Text style={[styles.sectionTitle, { color: c.text }]}>Why Choose WealthArena?</Text>
        <View style={styles.featuresGrid}>
          {features.map((feature, index) => (
            <View key={index} style={[styles.featureCard, { backgroundColor: c.surface }]}>
              <View style={[styles.featureIcon, { backgroundColor: feature.color + '20' }]}>
                <Icon name={feature.icon} size={32} color={feature.color} />
              </View>
              <Text style={[styles.featureTitle, { color: c.text }]}>{feature.title}</Text>
              <Text style={[styles.featureDescription, { color: c.textMuted }]}>{feature.description}</Text>
            </View>
          ))}
        </View>
      </View>

      {/* Screenshots Section */}
      <View style={styles.screenshotsSection}>
        <Text style={[styles.sectionTitle, { color: c.text }]}>See It In Action</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.screenshotsContainer}>
          {screenshots.map((screenshot, index) => (
            <View key={index} style={[styles.screenshotCard, { backgroundColor: c.surface }]}>
              <View style={[styles.screenshotPlaceholder, { backgroundColor: c.border }]}>
                <Icon name="monitor" size={48} color={c.textMuted} />
              </View>
              <Text style={[styles.screenshotTitle, { color: c.text }]}>{screenshot.title}</Text>
              <Text style={[styles.screenshotDescription, { color: c.textMuted }]}>{screenshot.description}</Text>
            </View>
          ))}
        </ScrollView>
      </View>

      {/* Value Proposition */}
      <View style={styles.valueSection}>
        <Text style={[styles.sectionTitle, { color: c.text }]}>Built for Every Investor</Text>
        <View style={styles.valueGrid}>
          <View style={[styles.valueCard, { backgroundColor: c.surface }]}>
            <Icon name="school" size={40} color={c.primary} />
            <Text style={[styles.valueTitle, { color: c.text }]}>Beginners</Text>
            <Text style={[styles.valueDescription, { color: c.textMuted }]}>
              Guided investment flows with educational content and AI assistance
            </Text>
          </View>
          <View style={[styles.valueCard, { backgroundColor: c.surface }]}>
            <Icon name="chart-bar" size={40} color={c.success} />
            <Text style={[styles.valueTitle, { color: c.text }]}>Advanced</Text>
            <Text style={[styles.valueDescription, { color: c.textMuted }]}>
              Full platform access with advanced analytics and strategy tools
            </Text>
          </View>
        </View>
      </View>

      {/* Authentication Options */}
      <View style={styles.authSection}>
        <Text style={[styles.sectionTitle, { color: c.text }]}>Get Started Today</Text>
        <Text style={[styles.authSubtitle, { color: c.textMuted }]}>
          Choose your preferred sign-up method
        </Text>
        
        <View style={styles.authButtons}>
          <TouchableOpacity 
            style={[styles.authButton, { backgroundColor: c.primary }]} 
            onPress={onNavigateToOnboarding}
          >
            <Icon name="email" size={20} color={c.background} />
            <Text style={[styles.authButtonText, { color: c.background }]}>Sign Up with Email</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={[styles.socialButton, { borderColor: c.border }]}>
            <Icon name="google" size={20} color="#DB4437" />
            <Text style={[styles.socialButtonText, { color: c.text }]}>Continue with Google</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={[styles.socialButton, { borderColor: c.border }]}>
            <Icon name="microsoft" size={20} color="#0078D4" />
            <Text style={[styles.socialButtonText, { color: c.text }]}>Continue with Microsoft</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={[styles.socialButton, { borderColor: c.border }]}>
            <Icon name="apple" size={20} color={c.text} />
            <Text style={[styles.socialButtonText, { color: c.text }]}>Continue with Apple</Text>
          </TouchableOpacity>
        </View>
        
        <View style={styles.authFooter}>
          <Text style={[styles.authFooterText, { color: c.textMuted }]}>
            Already have an account? 
          </Text>
          <TouchableOpacity onPress={onNavigateToAuth}>
            <Text style={[styles.authLink, { color: c.primary }]}>Sign In</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* CTA Section */}
      <View style={styles.ctaSection}>
        <Text style={[styles.ctaTitle, { color: c.text }]}>Ready to Start Your Journey?</Text>
        <Text style={[styles.ctaSubtitle, { color: c.textMuted }]}>
          Join thousands of investors who trust WealthArena for their financial growth
        </Text>
        <TouchableOpacity 
          style={[styles.ctaButton, { backgroundColor: c.primary }]} 
          onPress={onNavigateToOnboarding}
        >
          <Text style={[styles.ctaButtonText, { color: c.background }]}>Get Started Now</Text>
          <Icon name="rocket-launch" size={20} color={c.background} />
        </TouchableOpacity>
      </View>

      {/* Footer */}
      <View style={styles.footer}>
        <Text style={[styles.footerText, { color: c.textMuted }]}>
          © 2024 WealthArena. All rights reserved.
        </Text>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 24,
    paddingTop: 60,
    paddingBottom: 20,
  },
  logoContainer: {
    width: 48,
    height: 48,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  logoText: {
    fontSize: 20,
    fontWeight: '700',
    flex: 1,
  },
  loginButton: {
    borderWidth: 1,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
  },
  loginButtonText: {
    fontSize: 14,
    fontWeight: '600',
  },
  hero: {
    paddingHorizontal: 24,
    paddingVertical: 40,
    alignItems: 'center',
  },
  heroTitle: {
    fontSize: 32,
    fontWeight: '700',
    textAlign: 'center',
    marginBottom: 16,
  },
  heroSubtitle: {
    fontSize: 16,
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: 32,
    maxWidth: 320,
  },
  heroButtons: {
    flexDirection: 'row',
    gap: 16,
  },
  ctaButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 24,
    paddingVertical: 16,
    borderRadius: 12,
    gap: 8,
  },
  ctaButtonText: {
    fontSize: 16,
    fontWeight: '700',
  },
  secondaryButton: {
    borderWidth: 1,
    paddingHorizontal: 24,
    paddingVertical: 16,
    borderRadius: 12,
  },
  secondaryButtonText: {
    fontSize: 16,
    fontWeight: '600',
  },
  featuresSection: {
    paddingHorizontal: 24,
    paddingVertical: 40,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '700',
    textAlign: 'center',
    marginBottom: 32,
  },
  featuresGrid: {
    gap: 16,
  },
  featureCard: {
    padding: 20,
    borderRadius: 16,
    alignItems: 'center',
  },
  featureIcon: {
    width: 64,
    height: 64,
    borderRadius: 32,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  featureTitle: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 8,
    textAlign: 'center',
  },
  featureDescription: {
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 20,
  },
  screenshotsSection: {
    paddingVertical: 40,
  },
  screenshotsContainer: {
    paddingHorizontal: 24,
  },
  screenshotCard: {
    width: 280,
    marginRight: 16,
    padding: 20,
    borderRadius: 16,
    alignItems: 'center',
  },
  screenshotPlaceholder: {
    width: 240,
    height: 160,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  screenshotTitle: {
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 8,
    textAlign: 'center',
  },
  screenshotDescription: {
    fontSize: 14,
    textAlign: 'center',
  },
  valueSection: {
    paddingHorizontal: 24,
    paddingVertical: 40,
  },
  valueGrid: {
    gap: 16,
  },
  valueCard: {
    padding: 24,
    borderRadius: 16,
    alignItems: 'center',
  },
  valueTitle: {
    fontSize: 20,
    fontWeight: '700',
    marginTop: 16,
    marginBottom: 8,
  },
  valueDescription: {
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 20,
  },
  ctaSection: {
    paddingHorizontal: 24,
    paddingVertical: 40,
    alignItems: 'center',
  },
  ctaTitle: {
    fontSize: 28,
    fontWeight: '700',
    textAlign: 'center',
    marginBottom: 16,
  },
  ctaSubtitle: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 32,
    maxWidth: 320,
  },
  footer: {
    paddingHorizontal: 24,
    paddingVertical: 32,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 14,
  },
  headerButtons: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  authSection: {
    paddingHorizontal: 24,
    paddingVertical: 40,
    alignItems: 'center',
  },
  authSubtitle: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 32,
    maxWidth: 320,
  },
  authButtons: {
    width: '100%',
    gap: 12,
    marginBottom: 24,
  },
  authButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
    paddingVertical: 16,
    borderRadius: 12,
    gap: 12,
  },
  authButtonText: {
    fontSize: 16,
    fontWeight: '700',
  },
  socialButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
    paddingVertical: 16,
    borderRadius: 12,
    borderWidth: 1,
    gap: 12,
  },
  socialButtonText: {
    fontSize: 16,
    fontWeight: '600',
  },
  authFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  authFooterText: {
    fontSize: 14,
  },
  authLink: {
    fontSize: 14,
    fontWeight: '600',
  },
});

export default LandingScreen;
