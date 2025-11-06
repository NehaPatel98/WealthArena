/**
 * Onboarding Context
 * Manages dynamic AI-powered onboarding conversation state
 */

import React, { createContext, useContext, useState, useEffect, useMemo } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useUser } from './UserContext';
import { chatbotService } from '@/services/chatbotService';
import { apiService } from '@/services/apiService';

export interface OnboardingMessage {
  id: string;
  text: string;
  isBot: boolean;
  timestamp: Date;
  type?: 'question' | 'answer' | 'explanation' | 'celebration';
  questionType?: 'text' | 'choice' | 'multi' | 'slider' | 'timeline';
  options?: string[];
  mascotVariant?: 'excited' | 'learning' | 'confident' | 'cautious' | 'celebration';
}

export interface OnboardingAnswer {
  questionId: string;
  answer: string | string[];
  timestamp: Date;
  context?: string;
}

export interface OnboardingProfile {
  experienceLevel: 'beginner' | 'intermediate' | 'advanced';
  investmentGoals: string[];
  riskTolerance: 'conservative' | 'moderate' | 'aggressive' | 'very_aggressive';
  timeHorizon: 'short' | 'medium' | 'long' | 'very_long';
  learningStyle: 'hands-on' | 'structured' | 'exploratory';
  interestAreas: string[];
  selectedAvatar?: {
    type: 'mascot' | 'custom';
    variant?: string;
    url?: string;
  };
}

interface OnboardingContextType {
  // State
  sessionId: string | null;
  conversationHistory: OnboardingMessage[];
  currentQuestion: OnboardingMessage | null;
  userAnswers: OnboardingAnswer[];
  isLoading: boolean;
  isTyping: boolean;
  estimatedQuestionsRemaining: number;
  userProfile: Partial<OnboardingProfile>;
  stage: 'welcome' | 'questions' | 'avatar' | 'summary' | 'complete';
  
  // Progressive XP System
  unlockedFeatures: string[];
  nextUnlock: { feature: string; xpNeeded: number } | null;
  UNLOCK_THRESHOLDS: Record<string, number>;
  
  // Methods
  initializeOnboarding: () => Promise<void>;
  submitAnswer: (answer: string | string[]) => Promise<void>;
  skipQuestion: () => Promise<void>;
  goBack: () => Promise<void>;
  completeOnboarding: () => Promise<void>;
  saveProgress: () => Promise<void>;
  loadProgress: () => Promise<boolean>;
  setUserProfile: (profile: Partial<OnboardingProfile>) => void;
  setStage: (stage: 'welcome' | 'questions' | 'avatar' | 'summary' | 'complete') => void;
  getUnlockedFeatures: (xp: number) => string[];
  getNextUnlock: (xp: number) => { feature: string; xpNeeded: number } | null;
}

const OnboardingContext = createContext<OnboardingContextType | undefined>(undefined);

export function OnboardingProvider({ children }: Readonly<{ children: React.ReactNode }>) {
  const { user } = useUser();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [conversationHistory, setConversationHistory] = useState<OnboardingMessage[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState<OnboardingMessage | null>(null);
  const [userAnswers, setUserAnswers] = useState<OnboardingAnswer[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [estimatedQuestionsRemaining, setEstimatedQuestionsRemaining] = useState(5);
  const [userProfile, setUserProfile] = useState<Partial<OnboardingProfile>>({});
  const [stage, setStage] = useState<'welcome' | 'questions' | 'avatar' | 'summary' | 'complete'>('welcome');

  // Progressive XP System Constants
  const UNLOCK_THRESHOLDS = {
    dashboard: 0,
    learning: 0,
    game: 0,
    portfolio_builder: 100,
    analytics: 200,
    advanced_signals: 300,
    strategy_lab: 500,
  };

  // Calculate unlocked features based on XP
  const getUnlockedFeatures = (xp: number) => {
    return Object.entries(UNLOCK_THRESHOLDS)
      .filter(([_, threshold]) => xp >= threshold)
      .map(([feature]) => feature);
  };

  // Calculate next unlock milestone
  const getNextUnlock = (xp: number) => {
    const sortedFeatures = Object.entries(UNLOCK_THRESHOLDS)
      .sort(([_, thresholdA], [__, thresholdB]) => thresholdA - thresholdB)
      .filter(([_, threshold]) => xp < threshold);
    
    if (sortedFeatures.length === 0) return null;
    
    const [nextFeature, nextThreshold] = sortedFeatures[0];
    return {
      feature: nextFeature,
      xpNeeded: nextThreshold - xp
    };
  };

  // Get current unlocked features and next unlock
  const currentXP = user?.xp_points || 0;
  const unlockedFeatures = getUnlockedFeatures(currentXP);
  const nextUnlock = getNextUnlock(currentXP);

  // Save progress to AsyncStorage
  const saveProgress = async () => {
    try {
      const progressData = {
        sessionId,
        conversationHistory,
        userAnswers,
        userProfile,
        stage,
        timestamp: new Date().toISOString(),
      };
      await AsyncStorage.setItem('onboarding_progress', JSON.stringify(progressData));
    } catch (error) {
      console.error('Failed to save onboarding progress:', error);
    }
  };

  // Load progress from AsyncStorage
  const loadProgress = async (): Promise<boolean> => {
    try {
      const savedProgress = await AsyncStorage.getItem('onboarding_progress');
      if (savedProgress) {
        const progressData = JSON.parse(savedProgress);
        
        // Check if progress is recent (within 24 hours)
        const savedTime = new Date(progressData.timestamp);
        const now = new Date();
        const hoursDiff = (now.getTime() - savedTime.getTime()) / (1000 * 60 * 60);
        
        if (hoursDiff < 24) {
          setSessionId(progressData.sessionId);
          setConversationHistory(progressData.conversationHistory || []);
          setUserAnswers(progressData.userAnswers || []);
          setUserProfile(progressData.userProfile || {});
          setStage(progressData.stage || 'welcome');
          return true;
        } else {
          // Clear old progress
          await AsyncStorage.removeItem('onboarding_progress');
        }
      }
      return false;
    } catch (error) {
      console.error('Failed to load onboarding progress:', error);
      return false;
    }
  };

  // Initialize onboarding session
  const initializeOnboarding = async () => {
    if (!user) {
      console.warn('No user found, cannot initialize onboarding');
      return;
    }

    console.log('Initializing onboarding for user:', user.username);
    setIsLoading(true);
    
    try {
      // Try to load existing progress first
      const hasProgress = await loadProgress();
      if (hasProgress) {
        console.log('Found existing progress, resuming...');
        setStage('questions');
        setIsLoading(false);
        return;
      }

      // Check chatbot availability with timeout
      console.log('Checking chatbot availability...');
      const isChatbotAvailable = await Promise.race([
        chatbotService.healthCheck(),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), 5000))
      ]).catch(() => false);
      
      if (!isChatbotAvailable) {
        // Fallback to static onboarding
        console.warn('Chatbot unavailable, using static onboarding');
        await initializeStaticOnboarding();
        return;
      }

      // Start new session with AI
      const response = await chatbotService.startOnboarding({
        firstName: user.firstName || user.username,
        email: user.email,
        userId: user.user_id,
      });

      if (response.success) {
        setSessionId(response.sessionId);
        
        // Add welcome message
        const welcomeMessage: OnboardingMessage = {
          id: 'welcome-1',
          text: response.welcomeMessage || `Hi ${user.firstName || user.username}! 🎉 Welcome to WealthArena! I'm Foxy, your AI trading coach.`,
          isBot: true,
          timestamp: new Date(),
          type: 'question',
          mascotVariant: 'excited',
        };

        setConversationHistory([welcomeMessage]);
        setCurrentQuestion(welcomeMessage);
        setStage('questions');
        setEstimatedQuestionsRemaining(response.estimatedQuestions || 5);
        
        // Track onboarding start
        await trackAnalytics('onboarding_started', {
          mode: 'ai',
          estimatedQuestions: response.estimatedQuestions || 5,
        });
      } else {
        // Fallback to static onboarding
        await initializeStaticOnboarding();
      }
    } catch (error) {
      console.error('Failed to initialize onboarding:', error);
      // Fallback to static onboarding
      console.log('Falling back to static onboarding due to error');
      await initializeStaticOnboarding();
    } finally {
      console.log('Onboarding initialization complete, setting loading to false');
      setIsLoading(false);
    }
  };

  // Static onboarding fallback
  const initializeStaticOnboarding = async () => {
    console.log('Initializing static onboarding fallback');
    
    const staticQuestions = [
      {
        id: 'static-1',
        text: `Hi ${user?.firstName || user?.username}! 🎉 Welcome to WealthArena! Let's get to know you better.`,
        isBot: true,
        timestamp: new Date(),
        type: 'question' as const,
        questionType: 'text' as const,
        mascotVariant: 'excited' as const,
      },
      {
        id: 'static-2',
        text: 'What\'s your experience level with investing?',
        isBot: true,
        timestamp: new Date(),
        type: 'question' as const,
        questionType: 'choice' as const,
        options: ['Complete beginner', 'Some experience', 'Experienced'],
        mascotVariant: 'learning' as const,
      },
      {
        id: 'static-3',
        text: 'What\'s your primary investment goal?',
        isBot: true,
        timestamp: new Date(),
        type: 'question' as const,
        questionType: 'choice' as const,
        options: ['Learn the basics', 'Build wealth', 'Beat the market', 'Retirement planning'],
        mascotVariant: 'thinking' as const,
      },
      {
        id: 'static-4',
        text: 'How comfortable are you with investment risk?',
        isBot: true,
        timestamp: new Date(),
        type: 'question' as const,
        questionType: 'choice' as const,
        options: ['Very cautious', 'Moderate', 'Aggressive'],
        mascotVariant: 'confident' as const,
      },
    ];

    setConversationHistory(staticQuestions);
    setCurrentQuestion(staticQuestions[0]);
    setStage('questions');
    setEstimatedQuestionsRemaining(4);
    
    // Track static onboarding start
    await trackAnalytics('onboarding_started', {
      mode: 'static',
      estimatedQuestions: 4,
    });
  };

  // Submit answer and get next question
  const submitAnswer = async (answer: string | string[]) => {
    if (!currentQuestion) return;

    setIsLoading(true);
    setIsTyping(true);

    try {
      // Add user answer to history
      const userMessage: OnboardingMessage = {
        id: `user-${Date.now()}`,
        text: Array.isArray(answer) ? answer.join(', ') : answer,
        isBot: false,
        timestamp: new Date(),
        type: 'answer',
      };

      // Save answer
      const answerData: OnboardingAnswer = {
        questionId: currentQuestion.id,
        answer,
        timestamp: new Date(),
        context: currentQuestion.text,
      };

      setUserAnswers(prev => [...prev, answerData]);
      setConversationHistory(prev => [...prev, userMessage]);

      // Track answer submission
      await trackAnalytics('question_answered', {
        questionId: currentQuestion.id,
        questionType: currentQuestion.questionType,
        answerLength: Array.isArray(answer) ? answer.length : answer.length,
        timeSpent: Date.now() - (currentQuestion.timestamp?.getTime() || Date.now()),
      });

      // Handle static onboarding
      if (!sessionId || currentQuestion.id.startsWith('static-')) {
        await handleStaticOnboarding(answer);
        return;
      }

      // Get next question from AI
      const response = await chatbotService.sendOnboardingResponse(
        answer,
        {
          sessionId,
          conversationHistory: conversationHistory,
          userAnswers: [...userAnswers, answerData],
        }
      );

      if (response.success && response.nextQuestion) {
        const nextQuestion: OnboardingMessage = {
          id: response.nextQuestion.id,
          text: response.nextQuestion.text,
          isBot: true,
          timestamp: new Date(),
          type: 'question',
          questionType: response.nextQuestion.type,
          options: response.nextQuestion.options,
          mascotVariant: response.nextQuestion.mascotVariant,
        };

        setCurrentQuestion(nextQuestion);
        setConversationHistory(prev => [...prev, nextQuestion]);
        setEstimatedQuestionsRemaining(response.estimatedRemaining || Math.max(0, estimatedQuestionsRemaining - 1));

        // Update user profile based on AI analysis
        if (response.profileUpdates) {
          setUserProfile(prev => ({ ...prev, ...response.profileUpdates }));
        }
      } else if (response.complete) {
        // Onboarding complete, move to avatar selection
        setStage('avatar');
        await trackAnalytics('onboarding_questions_complete', {
          totalQuestions: conversationHistory.filter(msg => msg.isBot && msg.type === 'question').length,
          totalAnswers: userAnswers.length,
        });
      }
    } catch (error) {
      console.error('Failed to submit answer:', error);
      // Fallback to static onboarding
      await handleStaticOnboarding(answer);
    } finally {
      setIsLoading(false);
      setIsTyping(false);
      await saveProgress();
    }
  };

  // Handle static onboarding progression
  const handleStaticOnboarding = async (answer: string | string[]) => {
    const currentIndex = conversationHistory.findIndex(msg => msg.id === currentQuestion?.id);
    const nextIndex = currentIndex + 1;
    
    if (nextIndex < conversationHistory.length) {
      // Move to next static question
      const nextQuestion = conversationHistory[nextIndex];
      setCurrentQuestion(nextQuestion);
      setEstimatedQuestionsRemaining(prev => Math.max(0, prev - 1));
    } else {
      // Static onboarding complete, move to avatar selection
      setStage('avatar');
      await trackAnalytics('onboarding_questions_complete', {
        totalQuestions: conversationHistory.filter(msg => msg.isBot && msg.type === 'question').length,
        totalAnswers: userAnswers.length,
      });
    }
  };

  // Skip current question
  const skipQuestion = async () => {
    if (!currentQuestion) return;
    
    // Track question skip
    await trackAnalytics('question_skipped', {
      questionId: currentQuestion.id,
      questionType: currentQuestion.questionType,
    });
    
    await submitAnswer('skip');
  };

  // Go back to previous question
  const goBack = async () => {
    if (conversationHistory.length <= 1) return;

    // Track back navigation
    await trackAnalytics('question_back_navigation', {
      currentQuestionId: currentQuestion?.id,
      totalQuestions: conversationHistory.filter(msg => msg.isBot && msg.type === 'question').length,
    });

    // Remove last user answer and bot question
    const newHistory = conversationHistory.slice(0, -2);
    const newAnswers = userAnswers.slice(0, -1);
    
    setConversationHistory(newHistory);
    setUserAnswers(newAnswers);
    
    if (newHistory.length > 0) {
      const lastMessage = newHistory.at(-1);
      if (lastMessage?.isBot) {
        setCurrentQuestion(lastMessage);
      }
    }

    await saveProgress();
  };

  // Complete onboarding
  const completeOnboarding = async () => {
    if (!sessionId) return;

    setIsLoading(true);
    try {
      // First, get final profile from chatbot
      const chatbotResponse = await chatbotService.completeOnboarding({
        sessionId,
        conversationHistory,
        userAnswers,
        userProfile,
      });

      if (chatbotResponse.success) {
        // Then save to backend via apiService
        const backendData = await apiService.completeUserOnboarding({
          conversationHistory,
          userAnswers,
          userProfile: chatbotResponse.finalProfile || userProfile,
          selectedAvatar: userProfile.selectedAvatar,
        });
        
        setStage('complete');
        
        // Clear saved progress
        await AsyncStorage.removeItem('onboarding_progress');
        
        // Add completion message
        const completionMessage: OnboardingMessage = {
          id: 'complete-1',
          text: chatbotResponse.completionMessage || '🎉 Congratulations! Your profile is ready. Let\'s get you started!',
          isBot: true,
          timestamp: new Date(),
          type: 'celebration',
          mascotVariant: 'celebration',
        };

        setConversationHistory(prev => [...prev, completionMessage]);
        setCurrentQuestion(completionMessage);

        // Track onboarding completion
        await trackAnalytics('onboarding_completed', {
          totalTime: Date.now() - (sessionData?.startTime?.getTime() || Date.now()),
          totalQuestions: conversationHistory.filter(msg => msg.isBot && msg.type === 'question').length,
          totalAnswers: userAnswers.length,
          rewards: backendData.rewards,
          mode: sessionId ? 'ai' : 'static',
        });

        // Update user context with new profile data
        if (backendData.user) {
          // This would need to be implemented in UserContext
          // updateUser(backendData.user);
        }
      }
    } catch (error) {
      console.error('Failed to complete onboarding:', error);
      // Show error message to user
      const errorMessage: OnboardingMessage = {
        id: 'error-1',
        text: 'Sorry, there was an issue completing your profile. Please try again.',
        isBot: true,
        timestamp: new Date(),
        type: 'explanation',
        mascotVariant: 'cautious',
      };

      setConversationHistory(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Track analytics events
  const trackAnalytics = async (event: string, data?: any) => {
    try {
      const analyticsData = {
        event,
        userId: user?.user_id,
        sessionId,
        timestamp: new Date().toISOString(),
        stage,
        questionCount: conversationHistory.filter(msg => msg.isBot && msg.type === 'question').length,
        answerCount: userAnswers.length,
        ...data,
      };

      // Save to AsyncStorage for later sync
      const existingAnalytics = await AsyncStorage.getItem('onboarding_analytics');
      const analytics = existingAnalytics ? JSON.parse(existingAnalytics) : [];
      analytics.push(analyticsData);
      await AsyncStorage.setItem('onboarding_analytics', JSON.stringify(analytics));

      // Send to backend via apiService
      try {
        await apiService.trackOnboardingAnalytics(analyticsData);
      } catch (error) {
        // Analytics endpoint not available, continue silently
        console.log('Analytics endpoint not available');
      }
    } catch (error) {
      console.error('Failed to track analytics:', error);
    }
  };

  // Auto-save progress on changes
  useEffect(() => {
    if (sessionId && conversationHistory.length > 0) {
      saveProgress();
    }
  }, [sessionId, conversationHistory, userAnswers, userProfile, stage]);

  const contextValue = useMemo(
    () => ({
      sessionId,
      conversationHistory,
      currentQuestion,
      userAnswers,
      isLoading,
      isTyping,
      estimatedQuestionsRemaining,
      userProfile,
      stage,
      unlockedFeatures,
      nextUnlock,
      UNLOCK_THRESHOLDS,
      initializeOnboarding,
      submitAnswer,
      skipQuestion,
      goBack,
      completeOnboarding,
      saveProgress,
      loadProgress,
      setUserProfile,
      setStage,
      getUnlockedFeatures,
      getNextUnlock,
    }),
    [
      sessionId,
      conversationHistory,
      currentQuestion,
      userAnswers,
      isLoading,
      isTyping,
      estimatedQuestionsRemaining,
      userProfile,
      stage,
      unlockedFeatures,
      nextUnlock,
      UNLOCK_THRESHOLDS,
      getUnlockedFeatures,
      getNextUnlock,
    ]
  );

  return (
    <OnboardingContext.Provider value={contextValue}>
      {children}
    </OnboardingContext.Provider>
  );
}

export function useOnboarding() {
  const context = useContext(OnboardingContext);
  if (context === undefined) {
    throw new Error('useOnboarding must be used within an OnboardingProvider');
  }
  return context;
}
