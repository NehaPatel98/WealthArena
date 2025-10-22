import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { Stack } from 'expo-router';
import { BarChart3, TrendingUp, TrendingDown, Lock } from 'lucide-react-native';
import { useUserTier } from '@/contexts/UserTierContext';
import Colors from '@/constants/colors';

export default function AnalyticsScreen() {
  const { profile } = useUserTier();
  const isLocked = profile.tier === 'beginner';

  if (isLocked) {
    return (
      <View style={styles.container}>
        <Stack.Screen
          options={{
            title: 'Analytics',
            headerStyle: { backgroundColor: Colors.background },
            headerTintColor: Colors.text,
          }}
        />
        <View style={styles.lockedContainer}>
          <View style={styles.lockIcon}>
            <Lock size={48} color={Colors.textMuted} />
          </View>
          <Text style={styles.lockedTitle}>Intermediate Feature</Text>
          <Text style={styles.lockedDescription}>
            Upgrade to Intermediate tier to access advanced analytics and performance reports
          </Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Stack.Screen
        options={{
          title: 'Analytics',
          headerStyle: { backgroundColor: Colors.background },
          headerTintColor: Colors.text,
        }}
      />
      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <BarChart3 size={32} color={Colors.intermediate} />
          <Text style={styles.headerTitle}>Analytics Dashboard</Text>
          <Text style={styles.headerSubtitle}>Deep insights into your performance</Text>
        </View>

        <View style={styles.metricsGrid}>
          <View style={styles.metricCard}>
            <TrendingUp size={24} color={Colors.accent} />
            <Text style={styles.metricValue}>+18.5%</Text>
            <Text style={styles.metricLabel}>YTD Return</Text>
          </View>
          <View style={styles.metricCard}>
            <BarChart3 size={24} color={Colors.secondary} />
            <Text style={styles.metricValue}>1.82</Text>
            <Text style={styles.metricLabel}>Sharpe Ratio</Text>
          </View>
          <View style={styles.metricCard}>
            <TrendingDown size={24} color={Colors.danger} />
            <Text style={styles.metricValue}>-12.3%</Text>
            <Text style={styles.metricLabel}>Max Drawdown</Text>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Performance Attribution</Text>
          <View style={styles.attributionCard}>
            {[
              { factor: 'Stock Selection', contribution: '+8.2%', color: Colors.accent },
              { factor: 'Asset Allocation', contribution: '+5.1%', color: Colors.secondary },
              { factor: 'Market Timing', contribution: '+3.4%', color: Colors.gold },
              { factor: 'Sector Rotation', contribution: '+1.8%', color: Colors.intermediate },
            ].map((item) => (
              <View key={item.factor} style={styles.attributionRow}>
                <View style={styles.attributionLeft}>
                  <View style={[styles.attributionDot, { backgroundColor: item.color }]} />
                  <Text style={styles.attributionFactor}>{item.factor}</Text>
                </View>
                <Text style={[styles.attributionValue, { color: item.color }]}>
                  {item.contribution}
                </Text>
              </View>
            ))}
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Risk Metrics</Text>
          <View style={styles.riskCard}>
            {[
              { metric: 'Beta', value: '0.95' },
              { metric: 'Alpha', value: '+2.3%' },
              { metric: 'Volatility', value: '14.2%' },
              { metric: 'VaR (95%)', value: '-3.8%' },
            ].map((item) => (
              <View key={item.metric} style={styles.riskRow}>
                <Text style={styles.riskMetric}>{item.metric}</Text>
                <Text style={styles.riskValue}>{item.value}</Text>
              </View>
            ))}
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  lockedContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  lockIcon: {
    width: 96,
    height: 96,
    borderRadius: 48,
    backgroundColor: Colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
  },
  lockedTitle: {
    fontSize: 24,
    fontWeight: '700' as const,
    color: Colors.text,
    marginBottom: 12,
  },
  lockedDescription: {
    fontSize: 16,
    color: Colors.textSecondary,
    textAlign: 'center',
    lineHeight: 24,
  },
  content: {
    flex: 1,
  },
  header: {
    padding: 24,
    alignItems: 'center',
    gap: 12,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  headerSubtitle: {
    fontSize: 14,
    color: Colors.textSecondary,
    textAlign: 'center',
  },
  metricsGrid: {
    flexDirection: 'row',
    paddingHorizontal: 24,
    gap: 12,
    marginBottom: 32,
  },
  metricCard: {
    flex: 1,
    backgroundColor: Colors.surface,
    padding: 20,
    borderRadius: 16,
    alignItems: 'center',
    gap: 8,
  },
  metricValue: {
    fontSize: 20,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  metricLabel: {
    fontSize: 12,
    color: Colors.textSecondary,
    textAlign: 'center',
  },
  section: {
    paddingHorizontal: 24,
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '700' as const,
    color: Colors.text,
    marginBottom: 16,
  },
  attributionCard: {
    backgroundColor: Colors.surface,
    padding: 20,
    borderRadius: 16,
    gap: 16,
  },
  attributionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  attributionLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  attributionDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  attributionFactor: {
    fontSize: 14,
    color: Colors.text,
  },
  attributionValue: {
    fontSize: 16,
    fontWeight: '600' as const,
  },
  riskCard: {
    backgroundColor: Colors.surface,
    padding: 20,
    borderRadius: 16,
    gap: 16,
  },
  riskRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  riskMetric: {
    fontSize: 14,
    color: Colors.textSecondary,
  },
  riskValue: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text,
  },
});
