import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { colors } from '../theme/colors';

interface UserProfile {
  experience: 'beginner' | 'intermediate' | 'advanced';
  riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  investmentGoals: string[];
  timeHorizon: 'short' | 'medium' | 'long';
  suggestedPortfolio: string[];
}

interface Feature {
  id: string;
  name: string;
  description: string;
  icon: string;
  requiredLevel: 'beginner' | 'intermediate' | 'advanced';
  isUnlocked: boolean;
}

interface FeatureUnlockSystemProps {
  userProfile: UserProfile;
  onFeaturePress: (feature: Feature) => void;
}

const FeatureUnlockSystem: React.FC<FeatureUnlockSystemProps> = ({ userProfile, onFeaturePress }) => {
  const isDarkMode = true;
  const c = colors[isDarkMode ? 'dark' : 'light'];

  const features: Feature[] = [
    {
      id: 'portfolio-builder',
      name: 'Portfolio Builder',
      description: 'Create and manage your investment portfolio',
      icon: 'briefcase',
      requiredLevel: 'beginner',
      isUnlocked: true, // Always available
    },
    {
      id: 'guided-investing',
      name: 'Guided Investing',
      description: 'Step-by-step investment guidance',
      icon: 'school',
      requiredLevel: 'beginner',
      isUnlocked: userProfile.experience === 'beginner',
    },
    {
      id: 'ai-chatbot',
      name: 'AI Investment Advisor',
      description: 'Get personalized investment advice',
      icon: 'robot',
      requiredLevel: 'beginner',
      isUnlocked: true, // Always available
    },
    {
      id: 'basic-analytics',
      name: 'Basic Analytics',
      description: 'Simple portfolio performance metrics',
      icon: 'chart-line',
      requiredLevel: 'beginner',
      isUnlocked: userProfile.experience === 'beginner' || userProfile.experience === 'intermediate',
    },
    {
      id: 'advanced-analytics',
      name: 'Advanced Analytics',
      description: 'Detailed performance analysis and insights',
      icon: 'chart-bar',
      requiredLevel: 'intermediate',
      isUnlocked: userProfile.experience === 'intermediate' || userProfile.experience === 'advanced',
    },
    {
      id: 'strategy-lab',
      name: 'Strategy Lab',
      description: 'Test and optimize investment strategies',
      icon: 'flask',
      requiredLevel: 'intermediate',
      isUnlocked: userProfile.experience === 'intermediate' || userProfile.experience === 'advanced',
    },
    {
      id: 'risk-analysis',
      name: 'Risk Analysis',
      description: 'Advanced risk assessment tools',
      icon: 'shield-alert',
      requiredLevel: 'advanced',
      isUnlocked: userProfile.experience === 'advanced',
    },
    {
      id: 'algorithmic-trading',
      name: 'Algorithmic Trading',
      description: 'Automated trading strategies',
      icon: 'robot-industrial',
      requiredLevel: 'advanced',
      isUnlocked: userProfile.experience === 'advanced',
    },
  ];

  const getFeatureStatus = (feature: Feature) => {
    if (feature.isUnlocked) {
      return { color: c.success, text: 'Available' };
    } else {
      return { color: c.textMuted, text: 'Locked' };
    }
  };

  const getUpgradeMessage = () => {
    if (userProfile.experience === 'beginner') {
      return 'Complete more investments to unlock intermediate features';
    } else if (userProfile.experience === 'intermediate') {
      return 'Master advanced strategies to unlock expert features';
    }
    return 'You have access to all features';
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={[styles.title, { color: c.text }]}>Your Features</Text>
        <Text style={[styles.subtitle, { color: c.textMuted }]}>
          {userProfile.experience.charAt(0).toUpperCase() + userProfile.experience.slice(1)} Level
        </Text>
      </View>

      <View style={styles.featuresGrid}>
        {features.map((feature) => {
          const status = getFeatureStatus(feature);
          return (
            <TouchableOpacity
              key={feature.id}
              style={[
                styles.featureCard,
                { 
                  backgroundColor: c.surface,
                  borderColor: feature.isUnlocked ? c.primary : c.border,
                  opacity: feature.isUnlocked ? 1 : 0.6
                }
              ]}
              onPress={() => feature.isUnlocked && onFeaturePress(feature)}
              disabled={!feature.isUnlocked}
            >
              <View style={styles.featureHeader}>
                <View style={[
                  styles.featureIcon,
                  { backgroundColor: feature.isUnlocked ? c.primary : c.textMuted }
                ]}>
                  <Icon 
                    name={feature.isUnlocked ? feature.icon : 'lock'} 
                    size={24} 
                    color={c.background} 
                  />
                </View>
                <View style={styles.featureStatus}>
                  <Text style={[styles.statusText, { color: status.color }]}>
                    {status.text}
                  </Text>
                </View>
              </View>
              
              <Text style={[styles.featureName, { color: c.text }]}>
                {feature.name}
              </Text>
              
              <Text style={[styles.featureDescription, { color: c.textMuted }]}>
                {feature.description}
              </Text>

              {!feature.isUnlocked && (
                <View style={styles.lockedOverlay}>
                  <Text style={[styles.lockedText, { color: c.textMuted }]}>
                    Requires {feature.requiredLevel} level
                  </Text>
                </View>
              )}
            </TouchableOpacity>
          );
        })}
      </View>

      <View style={[styles.upgradeCard, { backgroundColor: c.surface }]}>
        <Icon name="trending-up" size={24} color={c.primary} />
        <View style={styles.upgradeContent}>
          <Text style={[styles.upgradeTitle, { color: c.text }]}>
            Unlock More Features
          </Text>
          <Text style={[styles.upgradeMessage, { color: c.textMuted }]}>
            {getUpgradeMessage()}
          </Text>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  header: {
    marginBottom: 24,
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
  },
  featuresGrid: {
    gap: 16,
    marginBottom: 24,
  },
  featureCard: {
    borderWidth: 1,
    borderRadius: 12,
    padding: 16,
    position: 'relative',
  },
  featureHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  featureIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  featureStatus: {
    alignItems: 'flex-end',
  },
  statusText: {
    fontSize: 12,
    fontWeight: '600',
  },
  featureName: {
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 4,
  },
  featureDescription: {
    fontSize: 14,
    lineHeight: 20,
  },
  lockedOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  lockedText: {
    fontSize: 12,
    fontWeight: '600',
  },
  upgradeCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 12,
    gap: 12,
  },
  upgradeContent: {
    flex: 1,
  },
  upgradeTitle: {
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 4,
  },
  upgradeMessage: {
    fontSize: 14,
    lineHeight: 20,
  },
});

export default FeatureUnlockSystem;
