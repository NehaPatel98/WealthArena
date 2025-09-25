import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Dimensions,
  ScrollView,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import AIChatbot from '../components/AIChatbot';
import { colors } from '../theme/colors';

interface UserProfile {
  experience: 'beginner' | 'intermediate' | 'advanced';
  riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  investmentGoals: string[];
  timeHorizon: 'short' | 'medium' | 'long';
  suggestedPortfolio: string[];
}

interface OnboardingScreenProps {
  onComplete: (userProfile?: UserProfile) => void;
}

const { width } = Dimensions.get('window');

const OnboardingScreen: React.FC<OnboardingScreenProps> = ({ onComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const isDarkMode = true;
  const c = colors[isDarkMode ? 'dark' : 'light'];

  const steps = [
    {
      icon: 'trending-up',
      title: 'Track Your Portfolio',
      description: 'Monitor your investments with real-time data and comprehensive analytics.',
      color: c.primary,
    },
    {
      icon: 'chart-line',
      title: 'Advanced Analytics',
      description: 'Get insights with detailed charts, performance metrics, and market analysis.',
      color: c.success,
    },
    {
      icon: 'shield-check',
      title: 'Secure & Private',
      description: 'Your financial data is protected with bank-level security and encryption.',
      color: c.warning,
    },
    {
      icon: 'robot',
      title: 'AI-Powered Insights',
      description: 'Get personalized investment recommendations from our AI chatbot.',
      color: c.accent,
    },
  ];

  const nextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      // Move to AI assessment
      setCurrentStep(4);
    }
  };

  const skipOnboarding = () => {
    onComplete();
  };

  const handleChatbotComplete = (profile: UserProfile) => {
    setUserProfile(profile);
    setCurrentStep(5); // Show results
  };

  const handleGetStarted = () => {
    onComplete(userProfile || undefined);
  };

  // Show AI Chatbot
  if (currentStep === 4) {
    return <AIChatbot onComplete={handleChatbotComplete} />;
  }

  // Show Results
  if (currentStep === 5 && userProfile) {
    return (
      <View style={[styles.container, { backgroundColor: c.background }]}>
        <ScrollView contentContainerStyle={styles.resultsContent} showsVerticalScrollIndicator={false}>
          <View style={styles.resultsContainer}>
            <View style={[styles.successIcon, { backgroundColor: c.success }]}>
              <Icon name="check" size={40} color={c.background} />
            </View>
            
            <Text style={[styles.resultsTitle, { color: c.text }]}>
              Your Investment Profile
            </Text>
            
            <View style={[styles.profileCard, { backgroundColor: c.surface }]}>
              <View style={styles.profileItem}>
                <Text style={[styles.profileLabel, { color: c.textMuted }]}>Experience Level</Text>
                <Text style={[styles.profileValue, { color: c.text }]}>{userProfile.experience}</Text>
              </View>
              
              <View style={styles.profileItem}>
                <Text style={[styles.profileLabel, { color: c.textMuted }]}>Risk Tolerance</Text>
                <Text style={[styles.profileValue, { color: c.text }]}>{userProfile.riskTolerance}</Text>
              </View>
              
              <View style={styles.profileItem}>
                <Text style={[styles.profileLabel, { color: c.textMuted }]}>Investment Goals</Text>
                <Text style={[styles.profileValue, { color: c.text }]}>{userProfile.investmentGoals.join(', ')}</Text>
              </View>
              
              <View style={styles.profileItem}>
                <Text style={[styles.profileLabel, { color: c.textMuted }]}>Time Horizon</Text>
                <Text style={[styles.profileValue, { color: c.text }]}>{userProfile.timeHorizon}</Text>
              </View>
            </View>

            <View style={[styles.portfolioCard, { backgroundColor: c.surface }]}>
              <Text style={[styles.portfolioTitle, { color: c.text }]}>Suggested Portfolio</Text>
              {userProfile.suggestedPortfolio.map((item, index) => (
                <View key={index} style={styles.portfolioItem}>
                  <View style={[styles.portfolioDot, { backgroundColor: c.primary }]} />
                  <Text style={[styles.portfolioText, { color: c.text }]}>{item}</Text>
                </View>
              ))}
            </View>

            <View style={[styles.featureUnlockCard, { backgroundColor: c.surface }]}>
              <Text style={[styles.unlockTitle, { color: c.text }]}>
                {userProfile.experience === 'beginner' ? 'Beginner Features Unlocked' : 'Full Platform Access'}
              </Text>
              <Text style={[styles.unlockDescription, { color: c.textMuted }]}>
                {userProfile.experience === 'beginner' 
                  ? 'You\'ll have access to guided investment flows, educational content, and AI assistance.'
                  : 'You have full access to advanced analytics, strategy tools, and all platform features.'
                }
              </Text>
            </View>
          </View>
        </ScrollView>

        <View style={styles.resultsNavigation}>
          <TouchableOpacity
            style={[styles.getStartedButton, { backgroundColor: c.primary }]}
            onPress={handleGetStarted}
          >
            <Text style={[styles.getStartedButtonText, { color: c.background }]}>
              Start Your Journey
            </Text>
            <Icon name="rocket-launch" size={20} color={c.background} />
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <View style={[styles.container, { backgroundColor: c.background }]}>
      {/* Progress Indicator */}
      <View style={styles.progressContainer}>
        <View style={styles.progressBar}>
          {steps.map((_, index) => (
            <View
              key={index}
              style={[
                styles.progressDot,
                {
                  backgroundColor: index <= currentStep ? c.primary : c.border,
                },
              ]}
            />
          ))}
        </View>
        <TouchableOpacity onPress={skipOnboarding} style={styles.skipButton}>
          <Text style={[styles.skipText, { color: c.textMuted }]}>Skip</Text>
        </TouchableOpacity>
      </View>

      {/* Content */}
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.stepContainer}>
          {/* Icon */}
          <View style={[styles.iconContainer, { backgroundColor: steps[currentStep].color + '20' }]}>
            <Icon
              name={steps[currentStep].icon}
              size={80}
              color={steps[currentStep].color}
            />
          </View>

          {/* Title */}
          <Text style={[styles.title, { color: c.text }]}>
            {steps[currentStep].title}
          </Text>

          {/* Description */}
          <Text style={[styles.description, { color: c.textMuted }]}>
            {steps[currentStep].description}
          </Text>
        </View>
      </ScrollView>

      {/* Navigation */}
      <View style={styles.navigation}>
        <TouchableOpacity
          style={[styles.nextButton, { backgroundColor: c.primary }]}
          onPress={nextStep}
        >
          <Text style={[styles.nextButtonText, { color: c.background }]}>
            {currentStep === steps.length - 1 ? 'Start Assessment' : 'Next'}
          </Text>
          <Icon
            name={currentStep === steps.length - 1 ? 'robot' : 'arrow-right'}
            size={20}
            color={c.background}
          />
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  progressContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 24,
    paddingTop: 60,
    paddingBottom: 20,
  },
  progressBar: {
    flexDirection: 'row',
    gap: 8,
  },
  progressDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  skipButton: {
    paddingVertical: 8,
    paddingHorizontal: 16,
  },
  skipText: {
    fontSize: 16,
    fontWeight: '600',
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 24,
  },
  stepContainer: {
    alignItems: 'center',
    maxWidth: 320,
  },
  iconContainer: {
    width: 160,
    height: 160,
    borderRadius: 80,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 32,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    textAlign: 'center',
    marginBottom: 16,
  },
  description: {
    fontSize: 16,
    lineHeight: 24,
    textAlign: 'center',
  },
  navigation: {
    paddingHorizontal: 24,
    paddingBottom: 40,
  },
  nextButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 12,
    paddingVertical: 16,
    gap: 8,
  },
  nextButtonText: {
    fontSize: 16,
    fontWeight: '700',
  },
  // Results styles
  resultsContent: {
    flexGrow: 1,
    paddingHorizontal: 24,
    paddingTop: 60,
    paddingBottom: 40,
  },
  resultsContainer: {
    alignItems: 'center',
  },
  successIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
  },
  resultsTitle: {
    fontSize: 28,
    fontWeight: '700',
    textAlign: 'center',
    marginBottom: 32,
  },
  profileCard: {
    width: '100%',
    padding: 20,
    borderRadius: 16,
    marginBottom: 20,
  },
  profileItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  profileLabel: {
    fontSize: 14,
    fontWeight: '600',
  },
  profileValue: {
    fontSize: 16,
    fontWeight: '700',
    textTransform: 'capitalize',
  },
  portfolioCard: {
    width: '100%',
    padding: 20,
    borderRadius: 16,
    marginBottom: 20,
  },
  portfolioTitle: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 16,
  },
  portfolioItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  portfolioDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 12,
  },
  portfolioText: {
    fontSize: 16,
    flex: 1,
  },
  featureUnlockCard: {
    width: '100%',
    padding: 20,
    borderRadius: 16,
    marginBottom: 32,
  },
  unlockTitle: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 8,
  },
  unlockDescription: {
    fontSize: 14,
    lineHeight: 20,
  },
  resultsNavigation: {
    paddingHorizontal: 24,
    paddingBottom: 40,
  },
  getStartedButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 12,
    paddingVertical: 16,
    gap: 8,
  },
  getStartedButtonText: {
    fontSize: 16,
    fontWeight: '700',
  },
});

export default OnboardingScreen;
