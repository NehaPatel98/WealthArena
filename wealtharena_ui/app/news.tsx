import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, Pressable, RefreshControl } from 'react-native';
import { useRouter, Stack } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { 
  useTheme, 
  Text, 
  Card, 
  Button, 
  Icon, 
  Badge,
  FAB,
  tokens 
} from '@/src/design-system';
import { newsService, NewsArticle } from '@/services/newsService';

export default function NewsScreen() {
  const router = useRouter();
  const { theme } = useTheme();
  const [news, setNews] = useState<NewsArticle[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Fetch news data
  const fetchNews = async () => {
    try {
      setIsLoading(true);
      const newsData = await newsService.getHighImpactNews();
      setNews(newsData);
    } catch (error) {
      console.error('Failed to fetch news:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Refresh news
  const onRefresh = async () => {
    setRefreshing(true);
    await fetchNews();
    setRefreshing(false);
  };

  useEffect(() => {
    fetchNews();
  }, []);

  const formatTime = (dateString: string) => {
    return new Date(dateString).toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric' 
    });
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
        <Text variant="h3" weight="semibold" style={styles.headerTitle}>Market News</Text>
        <View style={styles.headerRight} />
      </View>
      
      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={theme.primary}
            colors={[theme.primary]}
          />
        }
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.pageHeader}>
          <Text variant="h2" weight="bold">Market News</Text>
          <Text variant="body" muted>Stay updated with the latest market developments</Text>
        </View>

        {/* Loading State */}
        {isLoading ? (
          <View style={styles.loadingContainer}>
            <Text variant="body" muted>Loading news...</Text>
          </View>
        ) : (
          /* News List */
          <View style={styles.newsList}>
            {news.map((article, index) => (
              <Card key={article.id} style={styles.newsCard}>
                <Pressable
                  onPress={() => {
                    // Handle article press - could open in browser or show details
                    console.log('Article pressed:', article.title);
                  }}
                  style={styles.newsItem}
                >
                  {/* Article Header */}
                  <View style={styles.articleHeader}>
                    <View style={styles.articleMeta}>
                      <Text variant="small" muted>
                        {article.source} • {formatTime(article.publishedAt)}
                      </Text>
                      <Text variant="small" muted>
                        {formatDate(article.publishedAt)}
                      </Text>
                    </View>
                    {article.impact && (
                      <Badge 
                        variant={article.impact === 'high' ? 'danger' : 'secondary'} 
                        size="small"
                      >
                        {article.impact} impact
                      </Badge>
                    )}
                  </View>
                  
                  {/* Article Title */}
                  <Text variant="h4" weight="semibold" style={styles.articleTitle}>
                    {article.title}
                  </Text>
                  
                  {/* Article Summary */}
                  {article.summary && (
                    <Text variant="body" style={styles.articleSummary}>
                      {article.summary}
                    </Text>
                  )}
                  
                  {/* Article Footer */}
                  <View style={styles.articleFooter}>
                    <View style={styles.articleCategory}>
                      <Icon 
                        name={newsService.getCategoryIcon(article.category)} 
                        size={16} 
                        color={theme.primary} 
                      />
                      <Text variant="small" color={theme.primary} weight="medium">
                        {article.category}
                      </Text>
                    </View>
                    <Icon name="chevron-right" size={16} color={theme.muted} />
                  </View>
                </Pressable>
              </Card>
            ))}
          </View>
        )}

        {/* Empty State */}
        {!isLoading && news.length === 0 && (
          <View style={styles.emptyContainer}>
            <Icon name="newspaper" size={48} color={theme.muted} />
            <Text variant="h4" weight="semibold" style={styles.emptyTitle}>
              No News Available
            </Text>
            <Text variant="body" muted style={styles.emptyDescription}>
              Check back later for the latest market news and updates.
            </Text>
            <Button 
              variant="secondary" 
              size="medium" 
              onPress={fetchNews}
              style={styles.retryButton}
            >
              Try Again
            </Button>
          </View>
        )}

        {/* Bottom Spacing */}
        <View style={{ height: tokens.spacing.xl }} />
      </ScrollView>
      
      <FAB onPress={() => router.push('/ai-chat')} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
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
  scrollView: {
    flex: 1,
  },
  content: {
    padding: tokens.spacing.md,
  },
  pageHeader: {
    marginBottom: tokens.spacing.lg,
    gap: tokens.spacing.xs,
  },
  loadingContainer: {
    paddingVertical: tokens.spacing.xl,
    alignItems: 'center',
  },
  newsList: {
    gap: tokens.spacing.md,
  },
  newsCard: {
    padding: 0,
    overflow: 'hidden',
  },
  newsItem: {
    padding: tokens.spacing.md,
  },
  articleHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: tokens.spacing.sm,
  },
  articleMeta: {
    flex: 1,
    gap: 2,
  },
  articleTitle: {
    lineHeight: 24,
    marginBottom: tokens.spacing.sm,
  },
  articleSummary: {
    lineHeight: 20,
    marginBottom: tokens.spacing.md,
  },
  articleFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  articleCategory: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: tokens.spacing.xs,
  },
  emptyContainer: {
    alignItems: 'center',
    paddingVertical: tokens.spacing.xl,
    gap: tokens.spacing.md,
  },
  emptyTitle: {
    textAlign: 'center',
  },
  emptyDescription: {
    textAlign: 'center',
    maxWidth: 280,
  },
  retryButton: {
    marginTop: tokens.spacing.sm,
  },
});
