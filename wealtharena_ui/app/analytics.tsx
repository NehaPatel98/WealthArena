/**
 * Analytics Screen - Advanced Portfolio Analytics
 * Comprehensive portfolio performance analysis and visualizations
 */

import React, { useState, useEffect, useMemo } from 'react';
import { View, StyleSheet, ScrollView, Pressable, Dimensions } from 'react-native';
import { useRouter, Stack } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useUserTier } from '@/contexts/UserTierContext';
import { useTheme, Text, Card, Button, Badge, tokens, FAB } from '@/src/design-system';
import CandlestickChart from '../components/CandlestickChart';

const { width: screenWidth } = Dimensions.get('window');

interface CandleData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface PerformanceMetric {
  label: string;
  value: string;
  change?: string;
  changeType?: 'positive' | 'negative' | 'neutral';
  icon: string;
  color: string;
}

interface PortfolioData {
  totalValue: number;
  totalReturn: number;
  totalReturnPercent: number;
  dailyChange: number;
  dailyChangePercent: number;
  sharpeRatio: number;
  maxDrawdown: number;
  volatility: number;
  beta: number;
  alpha: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  profitFactor: number;
}

export default function AnalyticsScreen() {
  const router = useRouter();
  const { theme } = useTheme();
  const { profile, isLoading } = useUserTier();
  
  const [selectedTimeframe, setSelectedTimeframe] = useState<'1D' | '1W' | '1M' | '3M' | '1Y' | 'ALL'>('1M');
  const [selectedPortfolio, setSelectedPortfolio] = useState<string>('all');
  const [showAdvancedMetrics, setShowAdvancedMetrics] = useState(false);

  // Mock portfolio data
  const portfolioData: PortfolioData = {
    totalValue: 125750.50,
    totalReturn: 15750.50,
    totalReturnPercent: 14.32,
    dailyChange: 1250.25,
    dailyChangePercent: 1.00,
    sharpeRatio: 1.85,
    maxDrawdown: -8.5,
    volatility: 12.3,
    beta: 0.95,
    alpha: 2.1,
    winRate: 68.5,
    avgWin: 2.8,
    avgLoss: -1.2,
    profitFactor: 1.95,
  };

  // Mock performance data
  const performanceData: CandleData[] = useMemo(() => {
    const data: CandleData[] = [];
    const basePrice = 100000;
    const days = selectedTimeframe === '1D' ? 1 : 
                 selectedTimeframe === '1W' ? 7 :
                 selectedTimeframe === '1M' ? 30 :
                 selectedTimeframe === '3M' ? 90 :
                 selectedTimeframe === '1Y' ? 365 : 730;
    
    for (let i = 0; i < days; i++) {
      const date = new Date();
      date.setDate(date.getDate() - (days - i - 1));
      
      const randomChange = (Math.random() - 0.5) * 0.04; // ±2% daily change
      const price = basePrice * (1 + randomChange * i / days);
      const volatility = 0.02;
      
      const open = price * (1 + (Math.random() - 0.5) * volatility);
      const close = price * (1 + (Math.random() - 0.5) * volatility);
      const high = Math.max(open, close) * (1 + Math.random() * volatility);
      const low = Math.min(open, close) * (1 - Math.random() * volatility);
      
      data.push({
        timestamp: date.toISOString().split('T')[0],
        open: Math.round(open * 100) / 100,
        high: Math.round(high * 100) / 100,
        low: Math.round(low * 100) / 100,
        close: Math.round(close * 100) / 100,
        volume: Math.floor(Math.random() * 1000000) + 500000,
      });
    }
    
    return data;
  }, [selectedTimeframe]);

  // Performance metrics
  const performanceMetrics: PerformanceMetric[] = useMemo(() => [
    {
      label: 'Total Value',
      value: `$${portfolioData.totalValue.toLocaleString()}`,
      change: `+${portfolioData.totalReturnPercent.toFixed(2)}%`,
      changeType: 'positive',
      icon: 'trending-up',
      color: theme.success,
    },
    {
      label: 'Daily P&L',
      value: `$${portfolioData.dailyChange.toLocaleString()}`,
      change: `${portfolioData.dailyChangePercent > 0 ? '+' : ''}${portfolioData.dailyChangePercent.toFixed(2)}%`,
      changeType: portfolioData.dailyChange >= 0 ? 'positive' : 'negative',
      icon: 'pulse',
      color: portfolioData.dailyChange >= 0 ? theme.success : theme.danger,
    },
    {
      label: 'Sharpe Ratio',
      value: portfolioData.sharpeRatio.toFixed(2),
      change: 'Good',
      changeType: 'positive',
      icon: 'shield-checkmark',
      color: theme.primary,
    },
    {
      label: 'Max Drawdown',
      value: `${portfolioData.maxDrawdown.toFixed(1)}%`,
      change: 'Low Risk',
      changeType: 'positive',
      icon: 'trending-down',
      color: theme.warning,
    },
  ], [portfolioData, theme]);

  // Risk metrics
  const riskMetrics: PerformanceMetric[] = useMemo(() => [
    {
      label: 'Volatility',
      value: `${portfolioData.volatility.toFixed(1)}%`,
      change: 'Moderate',
      changeType: 'neutral',
      icon: 'stats-chart',
      color: theme.accent,
    },
    {
      label: 'Beta',
      value: portfolioData.beta.toFixed(2),
      change: 'Market Aligned',
      changeType: 'neutral',
      icon: 'analytics',
      color: theme.primary,
    },
    {
      label: 'Alpha',
      value: `+${portfolioData.alpha.toFixed(1)}%`,
      change: 'Outperforming',
      changeType: 'positive',
      icon: 'star',
      color: theme.success,
    },
    {
      label: 'Win Rate',
      value: `${portfolioData.winRate.toFixed(1)}%`,
      change: 'Excellent',
      changeType: 'positive',
      icon: 'trophy',
      color: theme.warning,
    },
  ], [portfolioData, theme]);

  // Trading metrics
  const tradingMetrics: PerformanceMetric[] = useMemo(() => [
    {
      label: 'Avg Win',
      value: `+${portfolioData.avgWin.toFixed(1)}%`,
      change: 'Good',
      changeType: 'positive',
      icon: 'arrow-up',
      color: theme.success,
    },
    {
      label: 'Avg Loss',
      value: `${portfolioData.avgLoss.toFixed(1)}%`,
      change: 'Controlled',
      changeType: 'positive',
      icon: 'arrow-down',
      color: theme.danger,
    },
    {
      label: 'Profit Factor',
      value: portfolioData.profitFactor.toFixed(2),
      change: 'Strong',
      changeType: 'positive',
      icon: 'calculator',
      color: theme.primary,
    },
    {
      label: 'Total Trades',
      value: '247',
      change: 'Active',
      changeType: 'neutral',
      icon: 'repeat',
      color: theme.accent,
    },
  ], [portfolioData, theme]);

  // Show loading state while profile is being loaded
  if (isLoading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: theme.bg }]} edges={['top']}>
        <Stack.Screen options={{ headerShown: false }} />
        <View style={[styles.header, { backgroundColor: theme.bg, borderBottomColor: theme.border }]}>
          <Pressable onPress={() => router.back()} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color={theme.text} />
          </Pressable>
          <Text variant="h3" weight="semibold" style={styles.headerTitle}>Analytics</Text>
          <View style={styles.headerRight} />
        </View>
        <View style={styles.loadingContainer}>
          <Ionicons name="analytics" size={64} color={theme.primary} />
          <Text variant="h3" weight="semibold" center>Loading Analytics...</Text>
        </View>
      </SafeAreaView>
    );
  }

  const isLocked = profile.tier === 'beginner' || profile.tier === null;

  if (isLocked) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: theme.bg }]} edges={['top']}>
        <Stack.Screen options={{ headerShown: false }} />
        <View style={[styles.header, { backgroundColor: theme.bg, borderBottomColor: theme.border }]}>
          <Pressable onPress={() => router.back()} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color={theme.text} />
          </Pressable>
          <Text variant="h3" weight="semibold" style={styles.headerTitle}>Analytics</Text>
          <View style={styles.headerRight} />
        </View>
        
        <View style={styles.lockedContainer}>
          <Card style={styles.lockedCard} elevation="med">
            <Ionicons name="lock-closed" size={64} color={theme.primary} />
            <Text variant="h2" weight="bold" center style={styles.lockedTitle}>
              Analytics Locked
            </Text>
            <Text variant="body" muted center style={styles.lockedDescription}>
              Advanced analytics are available for Intermediate and Expert users
            </Text>
            <Text variant="small" muted center style={styles.lockedSubtext}>
              Upgrade your account to access detailed portfolio analysis, risk metrics, and performance insights
            </Text>
            <Button
              variant="primary"
              size="large"
              onPress={() => router.push('/account')}
              icon={<Ionicons name="arrow-forward" size={20} color={theme.bg} />}
              style={styles.upgradeButton}
            >
              Upgrade Account
            </Button>
          </Card>
        </View>
      </SafeAreaView>
    );
  }

  const renderMetricCard = (metric: PerformanceMetric) => (
    <Card key={metric.label} style={styles.metricCard}>
      <View style={styles.metricHeader}>
        <Ionicons name={metric.icon as any} size={24} color={metric.color} />
        <Text variant="small" muted>{metric.label}</Text>
      </View>
      <Text variant="h3" weight="bold" style={[styles.metricValue, { color: metric.color }]}>
        {metric.value}
      </Text>
      {metric.change && (
        <View style={styles.metricChange}>
          <Text 
            variant="xs" 
            color={metric.changeType === 'positive' ? theme.success : 
                   metric.changeType === 'negative' ? theme.danger : theme.muted}
          >
            {metric.change}
          </Text>
        </View>
      )}
    </Card>
  );

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.bg }]} edges={['top']}>
      <Stack.Screen options={{ headerShown: false }} />
      
      {/* Custom Header */}
      <View style={[styles.header, { backgroundColor: theme.bg, borderBottomColor: theme.border }]}>
        <Pressable onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color={theme.text} />
        </Pressable>
        <Text variant="h3" weight="semibold" style={styles.headerTitle}>Analytics</Text>
        <View style={styles.headerRight}>
          <Pressable 
            onPress={() => setShowAdvancedMetrics(!showAdvancedMetrics)}
            style={styles.headerButton}
          >
            <Ionicons 
              name={showAdvancedMetrics ? "eye-off" : "eye"} 
              size={24} 
              color={theme.primary} 
            />
          </Pressable>
        </View>
      </View>

      <ScrollView style={styles.scrollView} contentContainerStyle={styles.content}>
        {/* Header Card */}
        <Card style={styles.headerCard} elevation="med" noBorder>
          <LinearGradient
            colors={[theme.primary, theme.accent]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.headerGradient}
          >
            <View style={styles.headerContent}>
              <Ionicons name="analytics" size={48} color={theme.bg} />
              <Text variant="h1" weight="bold" style={[styles.headerTitle, { color: theme.bg }]}>
                Portfolio Analytics
              </Text>
              <Text variant="body" style={[styles.headerSubtitle, { color: theme.bg + 'CC' }]}>
                Advanced performance analysis and risk metrics
              </Text>
            </View>
          </LinearGradient>
        </Card>

        {/* Timeframe Selector */}
        <Card style={styles.timeframeCard}>
          <Text variant="h3" weight="semibold" style={styles.sectionTitle}>Time Period</Text>
          <View style={styles.timeframeButtons}>
            {(['1D', '1W', '1M', '3M', '1Y', 'ALL'] as const).map((timeframe) => (
              <Pressable
                key={timeframe}
                onPress={() => setSelectedTimeframe(timeframe)}
                style={[
                  styles.timeframeButton,
                  selectedTimeframe === timeframe && { 
                    backgroundColor: theme.primary, 
                    borderColor: theme.primary 
                  }
                ]}
              >
                <Text 
                  variant="small" 
                  weight="semibold"
                  color={selectedTimeframe === timeframe ? theme.bg : theme.text}
                >
                  {timeframe}
                </Text>
              </Pressable>
            ))}
          </View>
        </Card>

        {/* Performance Chart */}
        <Card style={styles.chartCard}>
          <View style={styles.chartHeader}>
            <Text variant="h3" weight="semibold">Portfolio Performance</Text>
            <Badge variant="primary" size="small">{selectedTimeframe}</Badge>
          </View>
          
          {performanceData.length > 0 && (
            <CandlestickChart
              data={performanceData}
              chartType="daily"
              beginnerMode={false}
              showTooltip={true}
            />
          )}
          
          <View style={styles.chartStats}>
            <View style={styles.chartStat}>
              <Text variant="xs" muted>Start Value</Text>
              <Text variant="small" weight="semibold">
                ${performanceData[0]?.close.toLocaleString() || '0'}
              </Text>
            </View>
            <View style={styles.chartStat}>
              <Text variant="xs" muted>End Value</Text>
              <Text variant="small" weight="semibold">
                ${performanceData[performanceData.length - 1]?.close.toLocaleString() || '0'}
              </Text>
            </View>
            <View style={styles.chartStat}>
              <Text variant="xs" muted>Total Return</Text>
              <Text 
                variant="small" 
                weight="semibold"
                color={portfolioData.totalReturn >= 0 ? theme.success : theme.danger}
              >
                {portfolioData.totalReturn >= 0 ? '+' : ''}${portfolioData.totalReturn.toLocaleString()}
              </Text>
            </View>
          </View>
        </Card>

        {/* Key Performance Metrics */}
        <Card style={styles.metricsCard}>
          <Text variant="h3" weight="semibold" style={styles.sectionTitle}>Key Metrics</Text>
          <View style={styles.metricsGrid}>
            {performanceMetrics.map(renderMetricCard)}
          </View>
        </Card>

        {/* Risk Analysis */}
        <Card style={styles.metricsCard}>
          <Text variant="h3" weight="semibold" style={styles.sectionTitle}>Risk Analysis</Text>
          <View style={styles.metricsGrid}>
            {riskMetrics.map(renderMetricCard)}
          </View>
        </Card>

        {/* Advanced Trading Metrics */}
        {showAdvancedMetrics && (
          <Card style={styles.metricsCard}>
            <Text variant="h3" weight="semibold" style={styles.sectionTitle}>Trading Performance</Text>
            <View style={styles.metricsGrid}>
              {tradingMetrics.map(renderMetricCard)}
            </View>
          </Card>
        )}

        {/* Portfolio Allocation */}
        <Card style={styles.allocationCard}>
          <Text variant="h3" weight="semibold" style={styles.sectionTitle}>Portfolio Allocation</Text>
          <View style={styles.allocationChart}>
            <View style={styles.allocationItem}>
              <View style={styles.allocationBar}>
                <View style={[styles.allocationFill, { width: '35%', backgroundColor: theme.primary }]} />
              </View>
              <Text variant="small" weight="semibold">Stocks (35%)</Text>
            </View>
            <View style={styles.allocationItem}>
              <View style={styles.allocationBar}>
                <View style={[styles.allocationFill, { width: '25%', backgroundColor: theme.accent }]} />
              </View>
              <Text variant="small" weight="semibold">ETFs (25%)</Text>
            </View>
            <View style={styles.allocationItem}>
              <View style={styles.allocationBar}>
                <View style={[styles.allocationFill, { width: '20%', backgroundColor: theme.success }]} />
              </View>
              <Text variant="small" weight="semibold">Bonds (20%)</Text>
            </View>
            <View style={styles.allocationItem}>
              <View style={styles.allocationBar}>
                <View style={[styles.allocationFill, { width: '15%', backgroundColor: theme.warning }]} />
              </View>
              <Text variant="small" weight="semibold">Crypto (15%)</Text>
            </View>
            <View style={styles.allocationItem}>
              <View style={styles.allocationBar}>
                <View style={[styles.allocationFill, { width: '5%', backgroundColor: theme.danger }]} />
              </View>
              <Text variant="small" weight="semibold">Commodities (5%)</Text>
            </View>
          </View>
        </Card>

        {/* Performance Summary */}
        <Card style={styles.summaryCard}>
          <Text variant="h3" weight="semibold" style={styles.sectionTitle}>Performance Summary</Text>
          <View style={styles.summaryContent}>
            <View style={styles.summaryItem}>
              <Ionicons name="trending-up" size={24} color={theme.success} />
              <View style={styles.summaryText}>
                <Text variant="body" weight="semibold">Outperforming Market</Text>
                <Text variant="small" muted>Your portfolio is beating the S&P 500 by 2.1%</Text>
              </View>
            </View>
            <View style={styles.summaryItem}>
              <Ionicons name="shield-checkmark" size={24} color={theme.primary} />
              <View style={styles.summaryText}>
                <Text variant="body" weight="semibold">Risk Management</Text>
                <Text variant="small" muted>Low volatility with controlled drawdowns</Text>
              </View>
            </View>
            <View style={styles.summaryItem}>
              <Ionicons name="trophy" size={24} color={theme.warning} />
              <View style={styles.summaryText}>
                <Text variant="body" weight="semibold">Consistent Performance</Text>
                <Text variant="small" muted>68.5% win rate with strong profit factor</Text>
              </View>
            </View>
          </View>
        </Card>
      </ScrollView>

      {/* FAB */}
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
    flexDirection: 'row',
    alignItems: 'center',
    gap: tokens.spacing.sm,
  },
  headerButton: {
    padding: tokens.spacing.xs,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: tokens.spacing.md,
  },
  lockedContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: tokens.spacing.xl,
  },
  lockedCard: {
    alignItems: 'center',
    padding: tokens.spacing.xl,
    gap: tokens.spacing.md,
    maxWidth: 400,
  },
  lockedTitle: {
    marginTop: tokens.spacing.sm,
  },
  lockedDescription: {
    textAlign: 'center',
    marginBottom: tokens.spacing.sm,
  },
  lockedSubtext: {
    textAlign: 'center',
    marginBottom: tokens.spacing.lg,
  },
  upgradeButton: {
    marginTop: tokens.spacing.sm,
  },
  scrollView: { flex: 1 },
  content: {
    padding: tokens.spacing.md,
    gap: tokens.spacing.md,
    paddingBottom: tokens.spacing.xl * 2,
  },
  headerCard: {
    marginBottom: tokens.spacing.md,
  },
  headerGradient: {
    padding: tokens.spacing.xl,
    borderRadius: tokens.radius.lg,
  },
  headerContent: {
    alignItems: 'center',
    gap: tokens.spacing.sm,
  },
  headerTitleText: {
    textAlign: 'center',
  },
  headerSubtitle: {
    textAlign: 'center',
    marginTop: tokens.spacing.xs,
  },
  timeframeCard: {
    padding: tokens.spacing.md,
  },
  sectionTitle: {
    marginBottom: tokens.spacing.md,
  },
  timeframeButtons: {
    flexDirection: 'row',
    gap: tokens.spacing.sm,
  },
  timeframeButton: {
    flex: 1,
    paddingVertical: tokens.spacing.xs,
    paddingHorizontal: tokens.spacing.sm,
    borderRadius: tokens.radius.md,
    borderWidth: 1,
    borderColor: 'transparent',
    alignItems: 'center',
  },
  chartCard: {
    padding: tokens.spacing.md,
  },
  chartHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: tokens.spacing.md,
  },
  chartStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: tokens.spacing.md,
    paddingTop: tokens.spacing.md,
    borderTopWidth: 1,
    borderTopColor: 'rgba(0,0,0,0.1)',
  },
  chartStat: {
    alignItems: 'center',
    gap: tokens.spacing.xs,
  },
  metricsCard: {
    padding: tokens.spacing.md,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: tokens.spacing.sm,
  },
  metricCard: {
    flex: 1,
    minWidth: (screenWidth - tokens.spacing.md * 3) / 2,
    padding: tokens.spacing.md,
    alignItems: 'center',
    gap: tokens.spacing.xs,
  },
  metricHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: tokens.spacing.xs,
  },
  metricValue: {
    textAlign: 'center',
  },
  metricChange: {
    alignItems: 'center',
  },
  allocationCard: {
    padding: tokens.spacing.md,
  },
  allocationChart: {
    gap: tokens.spacing.sm,
  },
  allocationItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: tokens.spacing.md,
  },
  allocationBar: {
    flex: 1,
    height: 8,
    backgroundColor: 'rgba(0,0,0,0.1)',
    borderRadius: 4,
    overflow: 'hidden',
  },
  allocationFill: {
    height: '100%',
    borderRadius: 4,
  },
  summaryCard: {
    padding: tokens.spacing.md,
  },
  summaryContent: {
    gap: tokens.spacing.md,
  },
  summaryItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: tokens.spacing.md,
  },
  summaryText: {
    flex: 1,
  },
  fab: {
    position: 'absolute',
    bottom: tokens.spacing.xl,
    right: tokens.spacing.md,
  },
});