import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  TextInput,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { colors } from '../theme/colors';

interface Message {
  id: string;
  text: string;
  isBot: boolean;
  timestamp: Date;
}

interface AIChatbotProps {
  onComplete: (userProfile: UserProfile) => void;
}

interface UserProfile {
  experience: 'beginner' | 'intermediate' | 'advanced';
  riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  investmentGoals: string[];
  timeHorizon: 'short' | 'medium' | 'long';
  suggestedPortfolio: string[];
}

const AIChatbot: React.FC<AIChatbotProps> = ({ onComplete }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [userAnswers, setUserAnswers] = useState<Record<string, any>>({});
  const [isTyping, setIsTyping] = useState(false);
  const [inputText, setInputText] = useState('');
  const scrollViewRef = useRef<ScrollView>(null);
  
  const isDarkMode = true;
  const c = colors[isDarkMode ? 'dark' : 'light'];

  const questions = useMemo(() => [
    {
      id: 'experience',
      question: "Hi! I'm your AI investment advisor. Let's start with your experience level. How would you describe your investment knowledge?",
      options: [
        { text: 'Complete beginner - I\'m new to investing', value: 'beginner' },
        { text: 'Some experience - I\'ve made a few investments', value: 'intermediate' },
        { text: 'Experienced - I\'m comfortable with advanced strategies', value: 'advanced' }
      ],
      type: 'multiple_choice'
    },
    {
      id: 'riskTolerance',
      question: 'Great! Now, when it comes to risk, how do you feel about potential losses?',
      options: [
        { text: 'I prefer safe, stable investments', value: 'conservative' },
        { text: 'I can handle some ups and downs', value: 'moderate' },
        { text: 'I\'m comfortable with high-risk, high-reward', value: 'aggressive' }
      ],
      type: 'multiple_choice'
    },
    {
      id: 'investmentGoals',
      question: 'What are your main investment goals? (You can select multiple)',
      options: [
        { text: 'Retirement planning', value: 'retirement' },
        { text: 'Buying a home', value: 'home' },
        { text: 'Building emergency fund', value: 'emergency' },
        { text: 'Wealth building', value: 'wealth' },
        { text: 'Education funding', value: 'education' }
      ],
      type: 'multiple_select'
    },
    {
      id: 'timeHorizon',
      question: 'How long do you plan to invest before needing this money?',
      options: [
        { text: 'Less than 2 years', value: 'short' },
        { text: '2-10 years', value: 'medium' },
        { text: 'More than 10 years', value: 'long' }
      ],
      type: 'multiple_choice'
    },
    {
      id: 'investmentAmount',
      question: 'What amount are you comfortable starting with?',
      options: [
        { text: 'Under $1,000', value: 'small' },
        { text: '$1,000 - $10,000', value: 'medium' },
        { text: 'Over $10,000', value: 'large' }
      ],
      type: 'multiple_choice'
    },
    {
      id: 'marketKnowledge',
      question: 'How familiar are you with financial markets and economic concepts?',
      options: [
        { text: 'Not familiar - I need basic explanations', value: 'beginner' },
        { text: 'Somewhat familiar - I understand basic concepts', value: 'intermediate' },
        { text: 'Very familiar - I follow markets regularly', value: 'advanced' }
      ],
      type: 'multiple_choice'
    },
    {
      id: 'tradingExperience',
      question: 'Have you ever traded stocks, options, or other securities?',
      options: [
        { text: 'Never traded before', value: 'none' },
        { text: 'Traded a few times', value: 'limited' },
        { text: 'Regular trading experience', value: 'experienced' }
      ],
      type: 'multiple_choice'
    },
    {
      id: 'learningStyle',
      question: 'How do you prefer to learn about investing?',
      options: [
        { text: 'Guided tutorials and step-by-step instructions', value: 'guided' },
        { text: 'Interactive games and simulations', value: 'interactive' },
        { text: 'Advanced tools and analytics', value: 'advanced' }
      ],
      type: 'multiple_choice'
    }
  ], []);

  useEffect(() => {
    // Start the conversation
    setTimeout(() => {
      addBotMessage(questions[0].question);
    }, 1000);
  }, [questions]);

  const addBotMessage = (text: string) => {
    const message: Message = {
      id: Date.now().toString(),
      text,
      isBot: true,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, message]);
    setIsTyping(false);
  };

  const addUserMessage = (text: string) => {
    const message: Message = {
      id: Date.now().toString(),
      text,
      isBot: false,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, message]);
  };

  const handleAnswer = (questionId: string, answer: any) => {
    setUserAnswers(prev => ({ ...prev, [questionId]: answer }));
    
    // Add user message
    const currentQ = questions[currentQuestion];
    if (currentQ.type === 'multiple_choice') {
      const selectedOption = currentQ.options.find(opt => opt.value === answer);
      addUserMessage(selectedOption?.text || answer);
    } else if (currentQ.type === 'multiple_select') {
      const selectedOptions = currentQ.options.filter(opt => answer.includes(opt.value));
      addUserMessage(selectedOptions.map(opt => opt.text).join(', '));
    }

    // Move to next question or complete
    if (currentQuestion < questions.length - 1) {
      setIsTyping(true);
      setTimeout(() => {
        setCurrentQuestion(prev => prev + 1);
        addBotMessage(questions[currentQuestion + 1].question);
      }, 1000);
    } else {
      // Complete the assessment
      setIsTyping(true);
      setTimeout(() => {
        const userProfile = generateUserProfile();
        addBotMessage(`Perfect! Based on your answers, I've created a personalized investment profile for you. You're categorized as ${userProfile.experience} level with ${userProfile.riskTolerance} risk tolerance.`);
        setTimeout(() => {
          onComplete(userProfile);
        }, 2000);
      }, 1000);
    }
  };

  const generateUserProfile = (): UserProfile => {
    // Advanced tier categorization logic
    const experience = userAnswers.experience || 'beginner';
    const marketKnowledge = userAnswers.marketKnowledge || 'beginner';
    const tradingExperience = userAnswers.tradingExperience || 'none';
    const learningStyle = userAnswers.learningStyle || 'guided';
    
    // Calculate tier based on multiple factors
    let tierScore = 0;
    
    // Experience scoring
    if (experience === 'advanced') tierScore += 3;
    else if (experience === 'intermediate') tierScore += 2;
    else tierScore += 1;
    
    // Market knowledge scoring
    if (marketKnowledge === 'advanced') tierScore += 3;
    else if (marketKnowledge === 'intermediate') tierScore += 2;
    else tierScore += 1;
    
    // Trading experience scoring
    if (tradingExperience === 'experienced') tierScore += 3;
    else if (tradingExperience === 'limited') tierScore += 2;
    else tierScore += 1;
    
    // Learning style scoring
    if (learningStyle === 'advanced') tierScore += 2;
    else if (learningStyle === 'interactive') tierScore += 1;
    
    // Determine final tier
    let finalTier: 'beginner' | 'intermediate' | 'advanced';
    if (tierScore >= 8) finalTier = 'advanced';
    else if (tierScore >= 5) finalTier = 'intermediate';
    else finalTier = 'beginner';
    
    const riskTolerance = userAnswers.riskTolerance || 'moderate';
    const investmentGoals = userAnswers.investmentGoals || ['wealth'];
    const timeHorizon = userAnswers.timeHorizon || 'medium';

    // Generate suggested portfolio based on profile
    const suggestedPortfolio = generateSuggestedPortfolio(finalTier, riskTolerance, timeHorizon);

    return {
      experience: finalTier,
      riskTolerance,
      investmentGoals,
      timeHorizon,
      suggestedPortfolio
    };
  };

  const generateSuggestedPortfolio = (experience: string, riskTolerance: string, timeHorizon: string): string[] => {
    // Beginner portfolios
    if (experience === 'beginner') {
      if (riskTolerance === 'conservative') {
        return [
          'S&P 500 ETF (50%)',
          'Bond ETF (40%)',
          'Cash/Money Market (10%)'
        ];
      } else if (riskTolerance === 'moderate') {
        return [
          'S&P 500 ETF (60%)',
          'Bond ETF (30%)',
          'Cash/Money Market (10%)'
        ];
      } else {
        return [
          'S&P 500 ETF (70%)',
          'Bond ETF (20%)',
          'Cash/Money Market (10%)'
        ];
      }
    }
    
    // Intermediate portfolios
    if (experience === 'intermediate') {
      if (riskTolerance === 'conservative') {
        return [
          'S&P 500 ETF (45%)',
          'International ETF (15%)',
          'Bond ETF (30%)',
          'REIT ETF (10%)'
        ];
      } else if (riskTolerance === 'moderate') {
        return [
          'S&P 500 ETF (50%)',
          'International ETF (20%)',
          'Bond ETF (20%)',
          'REIT ETF (10%)'
        ];
      } else {
        return [
          'S&P 500 ETF (55%)',
          'International ETF (25%)',
          'Bond ETF (15%)',
          'REIT ETF (5%)'
        ];
      }
    }
    
    // Advanced portfolios
    if (experience === 'advanced') {
      if (riskTolerance === 'conservative') {
        return [
          'S&P 500 ETF (40%)',
          'International ETF (20%)',
          'Bond ETF (25%)',
          'REIT ETF (10%)',
          'Alternative Investments (5%)'
        ];
      } else if (riskTolerance === 'moderate') {
        return [
          'Growth Stocks (35%)',
          'S&P 500 ETF (30%)',
          'International ETF (20%)',
          'Bond ETF (10%)',
          'REIT ETF (5%)'
        ];
      } else {
        return [
          'Growth Stocks (40%)',
          'S&P 500 ETF (25%)',
          'International ETF (15%)',
          'Crypto ETF (10%)',
          'Bond ETF (5%)',
          'Alternative Investments (5%)'
        ];
      }
    }
    
    // Default fallback
    return ['S&P 500 ETF (60%)', 'Bond ETF (30%)', 'Cash/Money Market (10%)'];
  };

  const renderOptions = () => {
    const currentQ = questions[currentQuestion];
    if (!currentQ) return null;

    if (currentQ.type === 'multiple_choice') {
      return (
        <View style={styles.optionsContainer}>
          {currentQ.options.map((option, index) => (
            <TouchableOpacity
              key={index}
              style={[styles.optionButton, { borderColor: c.border }]}
              onPress={() => handleAnswer(currentQ.id, option.value)}
            >
              <Text style={[styles.optionText, { color: c.text }]}>{option.text}</Text>
            </TouchableOpacity>
          ))}
        </View>
      );
    }

    if (currentQ.type === 'multiple_select') {
      const selectedValues = userAnswers[currentQ.id] || [];
      return (
        <View style={styles.optionsContainer}>
          {currentQ.options.map((option, index) => {
            const isSelected = selectedValues.includes(option.value);
            return (
              <TouchableOpacity
                key={index}
                style={[
                  styles.optionButton,
                  { 
                    borderColor: isSelected ? c.primary : c.border,
                    backgroundColor: isSelected ? c.primary + '20' : 'transparent'
                  }
                ]}
                onPress={() => {
                  const newSelection = isSelected 
                    ? selectedValues.filter(v => v !== option.value)
                    : [...selectedValues, option.value];
                  setUserAnswers(prev => ({ ...prev, [currentQ.id]: newSelection }));
                }}
              >
                <Text style={[styles.optionText, { color: c.text }]}>{option.text}</Text>
                {isSelected && <Icon name="check" size={20} color={c.primary} />}
              </TouchableOpacity>
            );
          })}
          <TouchableOpacity
            style={[styles.continueButton, { backgroundColor: c.primary }]}
            onPress={() => handleAnswer(currentQ.id, userAnswers[currentQ.id] || [])}
          >
            <Text style={[styles.continueButtonText, { color: c.background }]}>Continue</Text>
          </TouchableOpacity>
        </View>
      );
    }

    return null;
  };

  return (
    <KeyboardAvoidingView 
      style={[styles.container, { backgroundColor: c.background }]}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      {/* Header */}
      <View style={[styles.header, { borderBottomColor: c.border }]}>
        <View style={styles.botInfo}>
          <View style={[styles.botAvatar, { backgroundColor: c.primary }]}>
            <Icon name="robot" size={24} color={c.background} />
          </View>
          <View>
            <Text style={[styles.botName, { color: c.text }]}>AI Advisor</Text>
            <Text style={[styles.botStatus, { color: c.textMuted }]}>Online</Text>
          </View>
        </View>
        <Text style={[styles.progress, { color: c.textMuted }]}>
          {currentQuestion + 1} of {questions.length}
        </Text>
      </View>

      {/* Messages */}
      <ScrollView 
        ref={scrollViewRef}
        style={styles.messagesContainer}
        contentContainerStyle={styles.messagesContent}
        onContentSizeChange={() => scrollViewRef.current?.scrollToEnd({ animated: true })}
      >
        {messages.map((message) => (
          <View key={message.id} style={[
            styles.messageContainer,
            message.isBot ? styles.botMessage : styles.userMessage
          ]}>
            <View style={[
              styles.messageBubble,
              message.isBot 
                ? { backgroundColor: c.surface } 
                : { backgroundColor: c.primary }
            ]}>
              <Text style={[
                styles.messageText,
                { color: message.isBot ? c.text : c.background }
              ]}>
                {message.text}
              </Text>
            </View>
          </View>
        ))}
        
        {isTyping && (
          <View style={styles.typingContainer}>
            <View style={[styles.typingBubble, { backgroundColor: c.surface }]}>
              <ActivityIndicator size="small" color={c.primary} />
              <Text style={[styles.typingText, { color: c.textMuted }]}>AI is thinking...</Text>
            </View>
          </View>
        )}
      </ScrollView>

      {/* Options */}
      {renderOptions()}
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
  },
  botInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  botAvatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  botName: {
    fontSize: 16,
    fontWeight: '700',
  },
  botStatus: {
    fontSize: 12,
  },
  progress: {
    fontSize: 14,
    fontWeight: '600',
  },
  messagesContainer: {
    flex: 1,
    paddingHorizontal: 20,
  },
  messagesContent: {
    paddingVertical: 20,
  },
  messageContainer: {
    marginBottom: 16,
  },
  botMessage: {
    alignItems: 'flex-start',
  },
  userMessage: {
    alignItems: 'flex-end',
  },
  messageBubble: {
    maxWidth: '80%',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 20,
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22,
  },
  typingContainer: {
    alignItems: 'flex-start',
    marginBottom: 16,
  },
  typingBubble: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 20,
    gap: 8,
  },
  typingText: {
    fontSize: 14,
  },
  optionsContainer: {
    padding: 20,
    gap: 12,
  },
  optionButton: {
    borderWidth: 1,
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  optionText: {
    fontSize: 16,
    flex: 1,
  },
  continueButton: {
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
    marginTop: 8,
  },
  continueButtonText: {
    fontSize: 16,
    fontWeight: '700',
  },
});

export default AIChatbot;
