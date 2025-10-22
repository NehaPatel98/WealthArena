import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { MessageCircle, Send, TrendingUp, Shield, Target } from 'lucide-react-native';
import { useUserTier } from '@/contexts/UserTierContext';
import Colors from '@/constants/colors';

interface Message {
  id: string;
  text: string;
  isBot: boolean;
}

const ONBOARDING_QUESTIONS = [
  {
    question: "Welcome to WealthArena! 👋 What's your name?",
    type: 'text' as const,
  },
  {
    question: "Great to meet you! Have you invested in stocks before?",
    type: 'choice' as const,
    options: ['Never', 'A little', 'Yes, regularly'],
  },
  {
    question: "What's your primary investment goal?",
    type: 'choice' as const,
    options: ['Learn the basics', 'Build wealth', 'Beat the market', 'Retirement planning'],
  },
  {
    question: "How comfortable are you with investment risk?",
    type: 'choice' as const,
    options: ['Very cautious', 'Moderate', 'Aggressive'],
  },
];

export default function OnboardingScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const { setUserTier, updateProfile } = useUserTier();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '0',
      text: ONBOARDING_QUESTIONS[0].question,
      isBot: true,
    },
  ]);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [inputText, setInputText] = useState('');
  const [userName, setUserName] = useState('');
  const [answers, setAnswers] = useState<string[]>([]);

  const scrollRef = useRef<ScrollView | null>(null);

  const addMessage = (text: string, isBot: boolean) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      text,
      isBot,
    };
    setMessages((prev) => [...prev, newMessage]);
  };

  useEffect(() => {
    try {
      scrollRef.current?.scrollToEnd({ animated: true });
    } catch (e) {
      console.log('scrollToEnd error', e);
    }
  }, [messages]);

  const handleAnswer = (answer: string) => {
    addMessage(answer, false);
    
    if (currentQuestion === 0) {
      setUserName(answer);
    }
    
    const newAnswers = [...answers, answer];
    setAnswers(newAnswers);

    setTimeout(() => {
      if (currentQuestion < ONBOARDING_QUESTIONS.length - 1) {
        const nextQuestion = currentQuestion + 1;
        setCurrentQuestion(nextQuestion);
        addMessage(ONBOARDING_QUESTIONS[nextQuestion].question, true);
      } else {
        finishOnboarding(newAnswers);
      }
    }, 500);
  };

  const finishOnboarding = (finalAnswers: string[]) => {
    const experienceAnswer = finalAnswers[1];
    const tier = experienceAnswer === 'Never' || experienceAnswer === 'A little' 
      ? 'beginner' 
      : 'intermediate';

    setTimeout(() => {
      addMessage(
        `Perfect! Based on your answers, I'm categorizing you as ${tier === 'beginner' ? 'a Beginner' : 'an Intermediate'} investor. Let's get started! 🚀`,
        true
      );

      setTimeout(() => {
        updateProfile({
          name: userName,
          tier,
        });
        setUserTier(tier);
        router.replace('/(tabs)/dashboard');
      }, 2000);
    }, 500);
  };

  const handleSendText = () => {
    if (inputText.trim()) {
      handleAnswer(inputText.trim());
      setInputText('');
    }
  };

  const currentQ = ONBOARDING_QUESTIONS[currentQuestion];
  const showInput = currentQ?.type === 'text';
  const showOptions = currentQ?.type === 'choice';

  return (
    <KeyboardAvoidingView
      style={[styles.container, { paddingTop: insets.top }]}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={0}
    >
      <View style={styles.header}>
        <View style={styles.headerIcon}>
          <TrendingUp size={28} color={Colors.gold} />
        </View>
        <Text style={styles.headerTitle}>WealthArena</Text>
        <Text style={styles.headerSubtitle}>AI-Powered Investment Coach</Text>
      </View>

      <ScrollView
        ref={scrollRef}
        style={styles.messagesContainer}
        contentContainerStyle={styles.messagesContent}
        showsVerticalScrollIndicator={false}
        onContentSizeChange={() => {
          try {
            scrollRef.current?.scrollToEnd({ animated: true });
          } catch (e) {
            console.log('onContentSizeChange scroll error', e);
          }
        }}
        testID="onboarding-scroll"
      >
        {messages.map((message) => (
          <View
            key={message.id}
            style={[
              styles.messageBubble,
              message.isBot ? styles.botMessage : styles.userMessage,
            ]}
          >
            {message.isBot && (
              <View style={styles.botIcon}>
                <MessageCircle size={16} color={Colors.secondary} />
              </View>
            )}
            <Text
              style={[
                styles.messageText,
                message.isBot ? styles.botMessageText : styles.userMessageText,
              ]}
            >
              {message.text}
            </Text>
          </View>
        ))}

        {showOptions && (
          <View style={styles.optionsContainer}>
            {currentQ.options?.map((option, index) => (
              <TouchableOpacity
                key={index}
                style={styles.optionButton}
                onPress={() => handleAnswer(option)}
              >
                <Text style={styles.optionText}>{option}</Text>
              </TouchableOpacity>
            ))}
          </View>
        )}
      </ScrollView>

      {showInput && (
        <View style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            value={inputText}
            onChangeText={setInputText}
            placeholder="Type your answer..."
            placeholderTextColor={Colors.textMuted}
            onSubmitEditing={handleSendText}
            returnKeyType="send"
          />
          <TouchableOpacity testID="onboarding-send" style={styles.sendButton} onPress={handleSendText}>
            <Send size={20} color={Colors.text} />
          </TouchableOpacity>
        </View>
      )}

      <View style={styles.features}>
        <View style={styles.feature}>
          <Shield size={16} color={Colors.accent} />
          <Text style={styles.featureText}>Secure</Text>
        </View>
        <View style={styles.feature}>
          <Target size={16} color={Colors.secondary} />
          <Text style={styles.featureText}>Personalized</Text>
        </View>
        <View style={styles.feature}>
          <TrendingUp size={16} color={Colors.gold} />
          <Text style={styles.featureText}>AI-Powered</Text>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  header: {
    paddingTop: 24,
    paddingBottom: 24,
    paddingHorizontal: 24,
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  headerIcon: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: Colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: '700' as const,
    color: Colors.text,
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 14,
    color: Colors.textSecondary,
  },
  messagesContainer: {
    flex: 1,
  },
  messagesContent: {
    padding: 24,
    gap: 16,
  },
  messageBubble: {
    maxWidth: '80%',
    padding: 16,
    borderRadius: 16,
  },
  botMessage: {
    alignSelf: 'flex-start',
    backgroundColor: Colors.surface,
    flexDirection: 'row',
    gap: 12,
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: Colors.secondary,
  },
  botIcon: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: Colors.surfaceLight,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 2,
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22,
  },
  botMessageText: {
    color: Colors.text,
    flex: 1,
  },
  userMessageText: {
    color: Colors.text,
  },
  optionsContainer: {
    gap: 12,
    marginTop: 8,
  },
  optionButton: {
    backgroundColor: Colors.surface,
    padding: 16,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: Colors.border,
  },
  optionText: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text,
    textAlign: 'center',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
    borderTopWidth: 1,
    borderTopColor: Colors.border,
    backgroundColor: Colors.surface,
  },
  input: {
    flex: 1,
    backgroundColor: Colors.surfaceLight,
    borderRadius: 24,
    paddingHorizontal: 20,
    paddingVertical: 12,
    fontSize: 16,
    color: Colors.text,
  },
  sendButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: Colors.secondary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  features: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 24,
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: Colors.border,
  },
  feature: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  featureText: {
    fontSize: 12,
    color: Colors.textSecondary,
    fontWeight: '600' as const,
  },
});
