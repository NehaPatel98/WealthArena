import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, Pressable, RefreshControl } from 'react-native';
import { useRouter, Stack } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useTheme, Text, Card, Button, Icon, Badge, ProgressRing, FAB, tokens } from '@/src/design-system';
import { apiService } from '@/services/apiService';
import { useGamification } from '@/contexts/GamificationContext';

const TOPICS = [
  { id: '1', title: 'Start Here', icon: 'trophy', completed: false, lessons: 5, progress: 0 },
  { id: '2', title: 'Investing Basics', icon: 'market', completed: false, lessons: 8, progress: 25 },
  { id: '3', title: 'Investing Strategies', icon: 'lab', completed: false, lessons: 12, progress: 0 },
  { id: '4', title: 'Portfolio Management', icon: 'portfolio', completed: false, lessons: 10, progress: 40 },
  { id: '5', title: 'Risk Analysis', icon: 'shield', completed: false, lessons: 7, progress: 0 },
  { id: '6', title: 'Technical Analysis', icon: 'signal', completed: false, lessons: 15, progress: 60 },
  { id: '7', title: 'Market Psychology', icon: 'agent', completed: false, lessons: 6, progress: 0 },
];

export default function LearningTopicsScreen() {
  const router = useRouter();
  const { theme } = useTheme();
  const { currentXP, currentLevel } = useGamification();
  const [topics, setTopics] = useState(TOPICS);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [learningProgress, setLearningProgress] = useState<any>(null);

  // Valid icon names from design-system Icon component
  const validIcons = [
    'market', 'signal', 'agent', 'replay', 'portfolio', 'shield', 'execute',
    'trophy', 'leaderboard', 'news', 'check-shield', 'lab', 'alert', 'settings',
    'send', 'google', 'bell', 'coin', 'xp', 'game-controller', 'robot'
  ];

  // Helper function to map category to icon (with validation)
  const getCategoryIcon = (category: string) => {
    const iconMap: { [key: string]: string } = {
      'Technical Analysis': 'signal',
      'Risk Management': 'shield',
      'Portfolio Management': 'portfolio',
      'Market Analysis': 'market',
      'Trading Strategies': 'lab',
      'Trading Psychology': 'agent',
      'Fundamentals': 'portfolio', // Changed from 'document-text' (doesn't exist)
      'Options': 'signal', // Changed from 'options' (doesn't exist)
      'Futures': 'signal', // Changed from 'time' (doesn't exist)  
      'Crypto': 'market', // Changed from 'logo-bitcoin' (doesn't exist)
    };
    
    const iconName = iconMap[category] || 'trophy';
    
    // Validate icon name against valid icons
    if (validIcons.includes(iconName)) {
      return iconName;
    }
    
    // Fallback to valid icon
    return 'trophy';
  };

  // Helper function to map difficulty to lesson count
  const getDifficultyLessons = (difficulty: string) => {
    const lessonMap: { [key: string]: number } = {
      'beginner': 5,
      'intermediate': 10,
      'advanced': 15,
    };
    return lessonMap[difficulty?.toLowerCase()] || 10;
  };

  // Load real learning data from backend
  useEffect(() => {
    const loadLearningData = async () => {
      try {
        setIsLoading(true);
        
        // Load knowledge topics from chatbot
        const topicsData = await apiService.getKnowledgeTopics();
        
        // Map chatbot response to UI format
        const mappedTopics = topicsData.map((topic: any) => ({
          id: topic.id,
          title: topic.title,
          icon: getCategoryIcon(topic.category),
          completed: false, // Will be updated from backend if available
          lessons: getDifficultyLessons(topic.difficulty),
          progress: 0 // Will be updated from backend if available
        }));
        
        setTopics(mappedTopics);
        
        // Load overall progress from backend
        try {
          const progressData = await apiService.getUserLearningProgress();
          setLearningProgress(progressData);
          
          // Merge progress into topics state
          if (progressData && progressData.topicsProgress) {
            const progressMap = new Map(
              (progressData.topicsProgress || []).map((p: any) => [p.topicId || p.topic_id, p])
            );
            
            const mappedTopicsWithProgress = mappedTopics.map(topic => {
              const progress = progressMap.get(topic.id);
              if (progress) {
                return {
                  ...topic,
                  progress: progress.progressPercent || progress.progress_percent || 0,
                  completed: progress.completed || progress.isCompleted || false,
                };
              }
              return topic;
            });
            
            setTopics(mappedTopicsWithProgress);
          } else {
            setTopics(mappedTopics);
          }
        } catch (progressError) {
          console.log('Learning progress not available yet');
          setTopics(mappedTopics);
        }
        
      } catch (error) {
        console.error('Failed to load learning data:', error);
        // Keep using mock data as fallback
      } finally {
        setIsLoading(false);
      }
    };

    loadLearningData();
  }, []);

  const onRefresh = async () => {
    setRefreshing(true);
    try {
      const topicsData = await apiService.getKnowledgeTopics();
      const mappedTopics = topicsData.map((topic: any) => ({
        id: topic.id,
        title: topic.title,
        icon: getCategoryIcon(topic.category),
        completed: false,
        lessons: getDifficultyLessons(topic.difficulty),
        progress: 0
      }));
      setTopics(mappedTopics);
      
      try {
        const progressData = await apiService.getUserLearningProgress();
        setLearningProgress(progressData);
        
        // Merge progress into topics state
        if (progressData && progressData.topicsProgress) {
          const progressMap = new Map(
            (progressData.topicsProgress || []).map((p: any) => [p.topicId || p.topic_id, p])
          );
          
          const mappedTopicsWithProgress = mappedTopics.map(topic => {
            const progress = progressMap.get(topic.id);
            if (progress) {
              return {
                ...topic,
                progress: progress.progressPercent || progress.progress_percent || 0,
                completed: progress.completed || progress.isCompleted || false,
              };
            }
            return topic;
          });
          
          setTopics(mappedTopicsWithProgress);
        } else {
          setTopics(mappedTopics);
        }
      } catch (progressError) {
        console.log('Learning progress not available yet');
        setTopics(mappedTopics);
      }
    } catch (error) {
      console.error('Failed to refresh learning data:', error);
    } finally {
      setRefreshing(false);
    }
  };

  const totalLessons = topics.reduce((sum, t) => sum + t.lessons, 0);
  const completedLessons = topics.reduce((sum, t) => sum + Math.floor(t.lessons * t.progress / 100), 0);
  const overallProgress = (completedLessons / totalLessons) * 100;

  // Handle topic selection - route to AI chat with learning mode
  const handleTopicPress = (topic: any) => {
    router.push(`/ai-chat?topic=${topic.id}&mode=learning`);
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.bg }]} edges={['top']}>
      <Stack.Screen
        options={{
          headerShown: false,
        }}
      />
      
      {/* Custom Header */}
      <View style={[styles.header, { backgroundColor: theme.bg, borderBottomColor: theme.border }]}>
        <Pressable onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color={theme.text} />
        </Pressable>
        <Text variant="h3" weight="semibold" style={styles.headerTitle}>Learning</Text>
        <View style={styles.headerRight} />
      </View>
      
      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {/* Header */}
        <Card style={styles.headerCard} elevation="med">
          <View style={styles.headerContent}>
            <View style={styles.headerLeft}>
              <Icon name="trophy" size={32} color={theme.yellow} />
              <View>
                <Text variant="h2" weight="bold">Learning Path</Text>
                <Text variant="small" muted>
                  {completedLessons} of {totalLessons} lessons completed
                </Text>
              </View>
            </View>
            <ProgressRing progress={overallProgress} size={60} showLabel={false} />
          </View>
        </Card>

        {/* Topics List */}
        {topics.map((topic) => (
          <Pressable 
            key={topic.id}
            onPress={() => handleTopicPress(topic)}
          >
            <Card style={styles.topicCard}>
              <View style={styles.topicHeader}>
                <View style={[styles.iconCircle, { backgroundColor: theme.primary + '20' }]}>
                  <Icon name={topic.icon as any} size={28} color={theme.primary} />
                </View>
                <View style={styles.topicInfo}>
                  <Text variant="body" weight="semibold">{topic.title}</Text>
                  <Text variant="small" muted>
                    {topic.lessons} lessons • {topic.progress}% complete
                  </Text>
                </View>
                {topic.completed && (
                  <Icon name="check-shield" size={24} color={theme.primary} />
                )}
              </View>

              {/* Progress Bar */}
              {topic.progress > 0 && (
                <View style={styles.progressContainer}>
                  <View style={[styles.progressBar, { backgroundColor: theme.border }]}>
                    <View 
                      style={[
                        styles.progressFill,
                        { backgroundColor: theme.primary, width: `${topic.progress}%` }
                      ]} 
                    />
                  </View>
                  <Text variant="xs" muted>{topic.progress}%</Text>
                </View>
              )}

              {topic.progress === 0 && (
                <Button variant="secondary" size="small">
                  Start Learning
                </Button>
              )}
            </Card>
          </Pressable>
        ))}

        <View style={{ height: 80 }} />
      </ScrollView>
      
      <FAB onPress={() => router.push('/ai-chat')} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: tokens.spacing.md,
    paddingVertical: tokens.spacing.sm,
    borderBottomWidth: 1,
    height: 56,
  },
  backButton: {
    padding: tokens.spacing.xs,
    marginLeft: -tokens.spacing.xs,
  },
  headerTitle: {
    flex: 1,
    textAlign: 'center',
    marginHorizontal: tokens.spacing.md,
  },
  headerRight: {
    width: 40,
  },
  scrollView: { flex: 1 },
  content: {
    padding: tokens.spacing.md,
    gap: tokens.spacing.md,
  },
  headerCard: {},
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: tokens.spacing.md,
    flex: 1,
  },
  topicCard: {
    gap: tokens.spacing.sm,
  },
  topicHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: tokens.spacing.sm,
  },
  iconCircle: {
    width: 52,
    height: 52,
    borderRadius: 26,
    alignItems: 'center',
    justifyContent: 'center',
  },
  topicInfo: {
    flex: 1,
    gap: 2,
  },
  progressContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: tokens.spacing.sm,
  },
  progressBar: {
    flex: 1,
    height: 8,
    borderRadius: tokens.radius.sm,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: tokens.radius.sm,
  },
});
