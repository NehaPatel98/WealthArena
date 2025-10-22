import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { Stack } from 'expo-router';
import { Target, TrendingUp, BarChart3, Lock } from 'lucide-react-native';
import { useUserTier } from '@/contexts/UserTierContext';
import Colors from '@/constants/colors';

const STRATEGIES = [
  { id: '1', name: 'Momentum Trading', description: 'Follow market trends', difficulty: 'Medium' },
  { id: '2', name: 'Mean Reversion', description: 'Buy low, sell high', difficulty: 'Hard' },
  { id: '3', name: 'Value Investing', description: 'Long-term fundamentals', difficulty: 'Easy' },
];

export default function StrategyLabScreen() {
  const { profile } = useUserTier();
  const isLocked = profile.tier === 'beginner';

  if (isLocked) {
    return (
      <View style={styles.container}>
        <Stack.Screen
          options={{
            title: 'Strategy Lab',
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
            Complete beginner challenges to unlock Strategy Lab and build custom trading strategies
          </Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Stack.Screen
        options={{
          title: 'Strategy Lab',
          headerStyle: { backgroundColor: Colors.background },
          headerTintColor: Colors.text,
        }}
      />
      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <Target size={32} color={Colors.accent} />
          <Text style={styles.headerTitle}>Strategy Lab</Text>
          <Text style={styles.headerSubtitle}>Build and test custom trading strategies</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Strategy Templates</Text>
          {STRATEGIES.map((strategy) => (
            <TouchableOpacity key={strategy.id} style={styles.strategyCard}>
              <View style={styles.strategyHeader}>
                <TrendingUp size={24} color={Colors.secondary} />
                <View style={styles.strategyInfo}>
                  <Text style={styles.strategyName}>{strategy.name}</Text>
                  <Text style={styles.strategyDescription}>{strategy.description}</Text>
                </View>
              </View>
              <View style={styles.difficultyBadge}>
                <Text style={styles.difficultyText}>{strategy.difficulty}</Text>
              </View>
            </TouchableOpacity>
          ))}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Backtest Results</Text>
          <View style={styles.resultsCard}>
            <BarChart3 size={32} color={Colors.gold} />
            <Text style={styles.resultsTitle}>No backtests yet</Text>
            <Text style={styles.resultsSubtitle}>Create a strategy to see results</Text>
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
  strategyCard: {
    backgroundColor: Colors.surface,
    padding: 20,
    borderRadius: 16,
    marginBottom: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  strategyHeader: {
    flexDirection: 'row',
    gap: 12,
    flex: 1,
  },
  strategyInfo: {
    flex: 1,
    gap: 4,
  },
  strategyName: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  strategyDescription: {
    fontSize: 14,
    color: Colors.textSecondary,
  },
  difficultyBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: Colors.surfaceLight,
    borderRadius: 12,
  },
  difficultyText: {
    fontSize: 12,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  resultsCard: {
    backgroundColor: Colors.surface,
    padding: 40,
    borderRadius: 16,
    alignItems: 'center',
    gap: 12,
  },
  resultsTitle: {
    fontSize: 18,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  resultsSubtitle: {
    fontSize: 14,
    color: Colors.textSecondary,
  },
});
