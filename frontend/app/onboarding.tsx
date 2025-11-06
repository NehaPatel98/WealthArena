import React, { useState, useEffect, useRef } from 'react';
import { View, StyleSheet, ScrollView, KeyboardAvoidingView, Platform, Animated, Dimensions } from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useUser } from '@/contexts/UserContext';
import { useOnboarding } from '@/contexts/OnboardingContext';
import { 
  useTheme, 
  Text, 
  Card, 
  Button, 
  TextInput, 
  tokens 
} from '@/src/design-system';
import CharacterMascot from '@/components/CharacterMascot';

const { width: screenWidth } = Dimensions.get('window');

// Helper function to map mascot variants to valid character types
const getMascotCharacter = (variant?: string): 'neutral' | 'excited' | 'confident' | 'thinking' | 'winner' | 'learning' | 'worried' | 'motivating' | 'happy' | 'sleeping' | 'cautious' | 'celebration' => {
  switch (variant) {
    case 'cautious':
      return 'worried';
    case 'celebration':
      return 'excited';
    case 'learning':
      return 'learning';
    case 'confident':
      return 'confident';
    case 'thinking':
      return 'thinking';
    default:
      return 'excited';
  }
};

export default function OnboardingScreen() {
  const router = useRouter();
  const { theme } = useTheme();
  const { user } = useUser();
  const {
    sessionId,
    conversationHistory,
    currentQuestion,
    isLoading,
    isTyping,
    estimatedQuestionsRemaining,
    userProfile,
    stage,
    initializeOnboarding,
    submitAnswer,
    skipQuestion,
    goBack,
    completeOnboarding,
    setUserProfile,
    setStage,
    unlockedFeatures,
    nextUnlock,
  } = useOnboarding();

  const [textInput, setTextInput] = useState('');
  const [selectedOptions, setSelectedOptions] = useState<string[]>([]);
  const [showCelebration, setShowCelebration] = useState(false);
  const scrollViewRef = useRef<ScrollView>(null);
  const mascotAnimation = useRef(new Animated.Value(0)).current;
  const typingAnimation = useRef(new Animated.Value(0)).current;
  const celebrationAnimation = useRef(new Animated.Value(0)).current;
  const confettiAnimation = useRef(new Animated.Value(0)).current;

  // Initialize onboarding on mount
  useEffect(() => {
    if (!sessionId && user) {
      initializeOnboarding();
    }
  }, [sessionId, user, initializeOnboarding]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (conversationHistory.length > 0) {
      setTimeout(() => {
        scrollViewRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [conversationHistory]);

  // Animate mascot based on typing state
  useEffect(() => {
    if (isTyping) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(mascotAnimation, {
            toValue: 1,
            duration: 500,
            useNativeDriver: true,
          }),
          Animated.timing(mascotAnimation, {
            toValue: 0,
            duration: 500,
            useNativeDriver: true,
          }),
        ])
      ).start();
    } else {
      mascotAnimation.setValue(0);
    }
  }, [isTyping, mascotAnimation]);

  // Animate typing indicator
  useEffect(() => {
    if (isTyping) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(typingAnimation, {
            toValue: 1,
            duration: 300,
            useNativeDriver: true,
          }),
          Animated.timing(typingAnimation, {
            toValue: 0,
            duration: 300,
            useNativeDriver: true,
          }),
        ])
      ).start();
    } else {
      typingAnimation.setValue(0);
    }
  }, [isTyping, typingAnimation]);

  // Celebration animations
  useEffect(() => {
    if (stage === 'complete') {
      setShowCelebration(true);
      
      // Start celebration sequence
      Animated.sequence([
        // Confetti burst
        Animated.timing(confettiAnimation, {
          toValue: 1,
          duration: 500,
          useNativeDriver: true,
        }),
        // Mascot celebration
        Animated.timing(celebrationAnimation, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
      ]).start();
    }
  }, [stage, celebrationAnimation, confettiAnimation]);

  const handleChoice = (choice: string) => {
    if (currentQuestion?.questionType === 'multi') {
      const newSelection = selectedOptions.includes(choice)
        ? selectedOptions.filter(opt => opt !== choice)
        : [...selectedOptions, choice];
      setSelectedOptions(newSelection);
    } else {
      submitAnswer(choice);
      setTextInput('');
    }
  };

  const handleMultiSubmit = () => {
    if (selectedOptions.length > 0) {
      submitAnswer(selectedOptions);
      setSelectedOptions([]);
    }
  };

  const handleTextSubmit = () => {
    if (!textInput.trim()) return;
    submitAnswer(textInput);
    setTextInput('');
  };

  const handleAvatarSelection = (variant: string) => {
    // Update user profile with selected avatar
    setUserProfile({
      ...userProfile,
      selectedAvatar: {
        type: 'mascot',
        variant: variant,
      }
    });
    
    // Move to completion stage
    setStage('complete');
  };

  const handleSkipAvatar = () => {
    // Use default avatar
    setUserProfile({
      ...userProfile,
      selectedAvatar: {
        type: 'mascot',
        variant: 'excited',
      }
    });
    
    // Move to completion stage
    setStage('complete');
  };

  const handleComplete = async () => {
    try {
      // Complete onboarding (backend handles XP/coin rewards via stored procedures)
      await completeOnboarding();
      
      // Navigate to dashboard after completion
      router.replace('/(tabs)/dashboard');
    } catch (error) {
      console.error('Failed to complete onboarding:', error);
      router.replace('/(tabs)/dashboard'); // Navigate anyway
    }
  };

  const handleSkip = () => {
    skipQuestion();
  };

  const handleBack = () => {
    goBack();
  };

  // Calculate progress based on conversation length
  const progress = conversationHistory.length > 0 
    ? Math.min(100, (conversationHistory.length / (conversationHistory.length + estimatedQuestionsRemaining)) * 100)
    : 0;

  // Show loading screen while initializing
  if (isLoading && !sessionId) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: theme.bg }]} edges={['top']}>
        <View style={styles.loadingContainer}>
          <Animated.View
            style={[
              styles.mascotContainer,
              {
                transform: [
                  {
                    scale: mascotAnimation.interpolate({
                      inputRange: [0, 1],
                      outputRange: [1, 1.1],
                    }),
                  },
                ],
              },
            ]}
          >
            <CharacterMascot character="excited" size={80} />
          </Animated.View>
          <Text variant="h3" style={styles.loadingText}>
            Setting up your personalized experience...
          </Text>
          <Text variant="small" style={[styles.loadingText, { marginTop: 16, opacity: 0.7 }]}>
            {user ? `Welcome, ${user.firstName || user.username}!` : 'Loading...'}
          </Text>
        </View>
      </SafeAreaView>
    );
  }

  // Safety fallback - if somehow we're not loading but have no content
  if (!isLoading && conversationHistory.length === 0 && !sessionId) {
    console.warn('Onboarding in unexpected state - no content, not loading, no session');
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: theme.bg }]} edges={['top']}>
        <View style={styles.loadingContainer}>
          <CharacterMascot character="worried" size={80} />
          <Text variant="h3" style={styles.loadingText}>
            Something went wrong
          </Text>
          <Text variant="body" style={[styles.loadingText, { marginTop: 16, opacity: 0.7 }]}>
            Let's try again
          </Text>
          <Button 
            variant="primary" 
            onPress={() => {
              console.log('Retrying onboarding initialization...');
              initializeOnboarding();
            }}
            style={{ marginTop: 24 }}
          >
            Retry
          </Button>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.bg }]} edges={['top']}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        {/* Header with Progress */}
        <View style={styles.header}>
          <View style={styles.progressContainer}>
            <View style={[styles.progressBg, { backgroundColor: theme.border }]}>
              <View 
                style={[
                  styles.progressFill, 
                  { backgroundColor: theme.primary, width: `${progress}%` }
                ]} 
              />
            </View>
            <Text variant="small" muted>
              {estimatedQuestionsRemaining > 0 
                ? `${estimatedQuestionsRemaining} questions remaining`
                : 'Almost done!'
              }
            </Text>
          </View>
        </View>

        {/* Chat Messages */}
        <ScrollView 
          ref={scrollViewRef}
          style={styles.chatContainer}
          contentContainerStyle={styles.chatContent}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {conversationHistory.map((message, index) => (
            <View key={message.id} style={styles.messageContainer}>
              {message.isBot ? (
                <View style={styles.botMessage}>
                  <View style={styles.botAvatar}>
                    <CharacterMascot 
                      character={getMascotCharacter(message.mascotVariant)} 
                      size={40} 
                    />
                  </View>
                  <Card style={styles.botBubble} elevation="low">
                    <Text variant="body" style={styles.botText}>
                      {message.text}
                    </Text>
                  </Card>
                </View>
              ) : (
                <View style={styles.userMessage}>
                  <Card style={styles.userBubble} elevation="low">
                    <Text variant="body" style={styles.userText}>
                      {message.text}
                    </Text>
                  </Card>
                </View>
              )}
            </View>
          ))}

          {/* Typing Indicator */}
          {isTyping && (
            <View style={styles.messageContainer}>
              <View style={styles.botMessage}>
                <View style={styles.botAvatar}>
                  <Animated.View
                    style={[
                      styles.typingMascot,
                      {
                        transform: [
                          {
                            scale: mascotAnimation.interpolate({
                              inputRange: [0, 1],
                              outputRange: [1, 1.05],
                            }),
                          },
                        ],
                      },
                    ]}
                  >
                    <CharacterMascot character="learning" size={40} />
                  </Animated.View>
                </View>
                <Card style={styles.botBubble} elevation="low">
                  <View style={styles.typingContainer}>
                    <Animated.View
                      style={[
                        styles.typingDot,
                        {
                          opacity: typingAnimation.interpolate({
                            inputRange: [0, 1],
                            outputRange: [0.3, 1],
                          }),
                        },
                      ]}
                    />
                    <Animated.View
                      style={[
                        styles.typingDot,
                        {
                          opacity: typingAnimation.interpolate({
                            inputRange: [0, 1],
                            outputRange: [0.3, 1],
                          }),
                        },
                      ]}
                    />
                    <Animated.View
                      style={[
                        styles.typingDot,
                        {
                          opacity: typingAnimation.interpolate({
                            inputRange: [0, 1],
                            outputRange: [0.3, 1],
                          }),
                        },
                      ]}
                    />
                  </View>
                </Card>
              </View>
            </View>
          )}
        </ScrollView>

        {/* Input Section */}
        {currentQuestion && !isTyping && (
          <View style={styles.inputSection}>
            {/* Choice Options */}
            {currentQuestion.questionType === 'choice' && currentQuestion.options && (
              <View style={styles.optionsContainer}>
                {currentQuestion.options.map((option) => (
                  <Button
                    key={option}
                    variant="secondary"
                    size="medium"
                    onPress={() => handleChoice(option)}
                    fullWidth
                    style={styles.optionButton}
                  >
                    {option}
                  </Button>
                ))}
              </View>
            )}

            {/* Multi-select Options */}
            {currentQuestion.questionType === 'multi' && currentQuestion.options && (
              <View style={styles.optionsContainer}>
                {currentQuestion.options.map((option) => (
                  <Button
                    key={option}
                    variant={selectedOptions.includes(option) ? "primary" : "secondary"}
                    size="medium"
                    onPress={() => handleChoice(option)}
                    fullWidth
                    style={styles.optionButton}
                  >
                    {option}
                  </Button>
                ))}
                {selectedOptions.length > 0 && (
                  <Button
                    variant="primary"
                    size="large"
                    onPress={handleMultiSubmit}
                    fullWidth
                    style={styles.submitButton}
                  >
                    Continue ({selectedOptions.length} selected)
                  </Button>
                )}
              </View>
            )}

            {/* Text Input */}
            {currentQuestion.questionType === 'text' && (
              <View style={styles.textInputContainer}>
                <TextInput
                  placeholder="Type your answer here..."
                  value={textInput}
                  onChangeText={setTextInput}
                  autoCapitalize="words"
                  onSubmitEditing={handleTextSubmit}
                  returnKeyType="done"
                  style={styles.textInput}
                />
                <Button
                  variant="primary"
                  size="medium"
                  onPress={handleTextSubmit}
                  disabled={!textInput.trim()}
                  style={styles.submitButton}
                >
                  Send
                </Button>
              </View>
            )}

            {/* Action Buttons */}
            <View style={styles.actionButtons}>
              {conversationHistory.length > 1 && (
                <Button
                  variant="ghost"
                  size="small"
                  onPress={handleBack}
                  style={styles.backButton}
                >
                  ← Back
                </Button>
              )}
              
              {currentQuestion && (
                <Button
                  variant="ghost"
                  size="small"
                  onPress={handleSkip}
                  style={styles.skipButton}
                >
                  Skip →
                </Button>
              )}
            </View>
          </View>
        )}

        {/* Avatar Selection Screen */}
        {stage === 'avatar' && (
          <View style={styles.avatarSelectionContainer}>
            <Text variant="h2" style={styles.avatarTitle}>
              Choose Your Trading Companion
            </Text>
            <Text variant="body" style={styles.avatarSubtitle}>
              Based on your personality, I recommend these avatars:
            </Text>
            
            <View style={styles.avatarGrid}>
              {[
                { variant: 'excited', name: 'The Optimist', description: 'Always ready for new opportunities' },
                { variant: 'confident', name: 'The Strategist', description: 'Calculated and methodical' },
                { variant: 'learning', name: 'The Scholar', description: 'Curious and analytical' },
                { variant: 'thinking', name: 'The Analyst', description: 'Data-driven and precise' },
              ].map((avatar) => (
                <Button
                  key={avatar.variant}
                  variant="secondary"
                  size="large"
                  onPress={() => handleAvatarSelection(avatar.variant)}
                  style={styles.avatarOption}
                >
                  <View style={styles.avatarPreview}>
                    <CharacterMascot character={avatar.variant as any} size={60} />
                    <Text variant="small" style={styles.avatarName}>
                      {avatar.name}
                    </Text>
                    <Text variant="xs" style={styles.avatarDescription}>
                      {avatar.description}
                    </Text>
                  </View>
                </Button>
              ))}
            </View>

            <Button
              variant="ghost"
              size="medium"
              onPress={handleSkipAvatar}
              style={styles.skipAvatarButton}
            >
              I'll choose later
            </Button>
          </View>
        )}

        {/* Completion Screen */}
        {stage === 'complete' && (
          <View style={styles.completionContainer}>
            {/* Confetti Effect */}
            {showCelebration && (
              <Animated.View
                style={[
                  styles.confettiContainer,
                  {
                    opacity: confettiAnimation.interpolate({
                      inputRange: [0, 1],
                      outputRange: [0, 1],
                    }),
                  },
                ]}
              >
                {new Array(20).fill(null).map((_, i) => {
                  const uniqueId = Math.random().toString(36).substr(2, 9);
                  return (
                  <Animated.View
                    key={`confetti-${uniqueId}`}
                    style={[
                      styles.confetti,
                      {
                        backgroundColor: ['#58CC02', '#1CB0F6', '#FFC800', '#FF4B4B'][i % 4],
                        transform: [
                          {
                            translateY: confettiAnimation.interpolate({
                              inputRange: [0, 1],
                              outputRange: [0, -200 - Math.random() * 100],
                            }),
                          },
                          {
                            translateX: confettiAnimation.interpolate({
                              inputRange: [0, 1],
                              outputRange: [0, (Math.random() - 0.5) * 200],
                            }),
                          },
                        ],
                      },
                    ]}
                  />
                  );
                })}
              </Animated.View>
            )}

            <Animated.View
              style={[
                styles.celebrationMascot,
                {
                  transform: [
                    {
                      scale: celebrationAnimation.interpolate({
                        inputRange: [0, 1],
                        outputRange: [0.8, 1.2],
                      }),
                    },
                    {
                      rotate: celebrationAnimation.interpolate({
                        inputRange: [0, 1],
                        outputRange: ['0deg', '5deg'],
                      }),
                    },
                  ],
                },
              ]}
            >
              <CharacterMascot character="excited" size={80} />
            </Animated.View>

            <Animated.View
              style={[
                styles.completionContent,
                {
                  opacity: celebrationAnimation,
                  transform: [
                    {
                      translateY: celebrationAnimation.interpolate({
                        inputRange: [0, 1],
                        outputRange: [20, 0],
                      }),
                    },
                  ],
                },
              ]}
            >
              <Text variant="h2" style={styles.completionTitle}>
                🎉 Welcome to WealthArena!
              </Text>
              <Text variant="body" style={styles.completionText}>
                Your profile is ready. Let's start your trading journey!
              </Text>

              {/* Rewards Display */}
              <View style={styles.rewardsContainer}>
                <View style={styles.rewardItem}>
                  <Text variant="h3" style={styles.rewardAmount}>+50 XP</Text>
                  <Text variant="small" style={styles.rewardLabel}>Welcome Bonus</Text>
                </View>
                <View style={styles.rewardItem}>
                  <Text variant="h3" style={styles.rewardAmount}>+500 Coins</Text>
                  <Text variant="small" style={styles.rewardLabel}>Starting Capital</Text>
                </View>
                <View style={styles.rewardItem}>
                  <Text variant="h3" style={styles.rewardAmount}>🏆</Text>
                  <Text variant="small" style={styles.rewardLabel}>First Achievement</Text>
                </View>
              </View>

              {/* Features Unlocked */}
              {unlockedFeatures && unlockedFeatures.length > 0 && (
                <View style={styles.unlockedContainer}>
                  <Text variant="small" weight="semibold" style={{ marginBottom: tokens.spacing.xs }}>
                    ✅ Features Unlocked
                  </Text>
                  <Text variant="xs" muted>
                    {unlockedFeatures.join(', ').replace(/_/g, ' ')}
                  </Text>
                  {nextUnlock && (
                    <Text variant="xs" style={{ marginTop: tokens.spacing.xs, color: theme.primary }}>
                      Next: Unlock {nextUnlock.feature.replace(/_/g, ' ')} at {nextUnlock.xpNeeded} more XP
                    </Text>
                  )}
                </View>
              )}

              <Button
                variant="primary"
                size="large"
                onPress={handleComplete}
                fullWidth
                style={styles.completeButton}
              >
                Get Started
              </Button>
            </Animated.View>
          </View>
        )}
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  keyboardView: {
    flex: 1,
  },
  
  // Loading Screen
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: tokens.spacing.xl,
    gap: tokens.spacing.lg,
  },
  mascotContainer: {
    alignItems: 'center',
  },
  loadingText: {
    textAlign: 'center',
    color: tokens.color.neutral500,
  },

  // Header
  header: {
    padding: tokens.spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: tokens.color.neutral200,
  },
  progressContainer: {
    gap: tokens.spacing.sm,
  },
  progressBg: {
    height: 8,
    borderRadius: tokens.radius.sm,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: tokens.radius.sm,
  },

  // Chat Interface
  chatContainer: {
    flex: 1,
  },
  chatContent: {
    padding: tokens.spacing.md,
    gap: tokens.spacing.md,
  },
  messageContainer: {
    marginBottom: tokens.spacing.sm,
  },
  
  // Bot Messages
  botMessage: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: tokens.spacing.sm,
  },
  botAvatar: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  botBubble: {
    flex: 1,
    maxWidth: screenWidth * 0.75,
    padding: tokens.spacing.md,
    backgroundColor: tokens.color.neutral200,
  },
  botText: {
    color: tokens.color.neutral900,
  },
  
  // User Messages
  userMessage: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
  },
  userBubble: {
    maxWidth: screenWidth * 0.75,
    padding: tokens.spacing.md,
    backgroundColor: tokens.color.primary,
  },
  userText: {
    color: tokens.color.white,
  },

  // Typing Indicator
  typingMascot: {
    alignItems: 'center',
  },
  typingContainer: {
    flexDirection: 'row',
    gap: tokens.spacing.xs,
    alignItems: 'center',
  },
  typingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: tokens.color.neutral500,
  },

  // Input Section
  inputSection: {
    padding: tokens.spacing.md,
    borderTopWidth: 1,
    borderTopColor: tokens.color.neutral200,
    backgroundColor: tokens.color.white,
  },
  optionsContainer: {
    gap: tokens.spacing.sm,
    marginBottom: tokens.spacing.md,
  },
  optionButton: {
    marginBottom: tokens.spacing.xs,
  },
  textInputContainer: {
    flexDirection: 'row',
    gap: tokens.spacing.sm,
    alignItems: 'flex-end',
  },
  textInput: {
    flex: 1,
  },
  submitButton: {
    minWidth: 80,
  },
  actionButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: tokens.spacing.sm,
  },
  backButton: {
    flex: 1,
    marginRight: tokens.spacing.sm,
  },
  skipButton: {
    flex: 1,
    marginLeft: tokens.spacing.sm,
  },

  // Completion Screen
  completionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: tokens.spacing.xl,
    gap: tokens.spacing.lg,
  },
  celebrationMascot: {
    alignItems: 'center',
  },
  completionTitle: {
    textAlign: 'center',
    color: tokens.color.primary,
  },
  completionText: {
    textAlign: 'center',
    color: tokens.color.neutral500,
    marginBottom: tokens.spacing.lg,
  },
  completeButton: {
    marginTop: tokens.spacing.md,
  },

  // Avatar Selection
  avatarSelectionContainer: {
    flex: 1,
    padding: tokens.spacing.lg,
    gap: tokens.spacing.lg,
  },
  avatarTitle: {
    textAlign: 'center',
    color: tokens.color.primary,
    marginBottom: tokens.spacing.sm,
  },
  avatarSubtitle: {
    textAlign: 'center',
    color: tokens.color.neutral500,
    marginBottom: tokens.spacing.lg,
  },
  avatarGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: tokens.spacing.md,
    justifyContent: 'space-between',
  },
  avatarOption: {
    width: '48%',
    padding: tokens.spacing.md,
    alignItems: 'center',
  },
  avatarPreview: {
    alignItems: 'center',
    gap: tokens.spacing.sm,
  },
  avatarName: {
    fontWeight: '600',
    color: tokens.color.neutral900,
  },
  avatarDescription: {
    textAlign: 'center',
    color: tokens.color.neutral500,
  },
  skipAvatarButton: {
    marginTop: tokens.spacing.lg,
    alignSelf: 'center',
  },

  // Celebration Effects
  confettiContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 1,
  },
  confetti: {
    position: 'absolute',
    width: 8,
    height: 8,
    borderRadius: 4,
    top: '50%',
    left: '50%',
  },
  completionContent: {
    alignItems: 'center',
    gap: tokens.spacing.lg,
  },
  rewardsContainer: {
    flexDirection: 'row',
    gap: tokens.spacing.md,
    marginVertical: tokens.spacing.lg,
  },
  rewardItem: {
    alignItems: 'center',
    padding: tokens.spacing.md,
    backgroundColor: tokens.color.neutral200,
    borderRadius: tokens.radius.md,
    minWidth: 80,
  },
  rewardAmount: {
    color: tokens.color.primary,
    fontWeight: '700',
  },
  rewardLabel: {
    color: tokens.color.neutral500,
    textAlign: 'center',
  },
  unlockedContainer: {
    marginTop: tokens.spacing.sm,
    padding: tokens.spacing.md,
    backgroundColor: tokens.color.neutral100,
    borderRadius: tokens.radius.md,
    width: '100%',
  },
});
