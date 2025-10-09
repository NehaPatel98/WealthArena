import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import { Gamepad2, Trophy, Clock, TrendingUp, Users, Play, Star, Zap } from 'lucide-react-native';
import Colors from '@/constants/colors';
import { LinearGradient } from 'expo-linear-gradient';

const { width } = Dimensions.get('window');

const SCENARIOS = [
  {
    id: '1',
    title: '2008 Financial Crisis',
    description: 'Navigate the market crash and recovery',
    difficulty: 'Hard',
    duration: '6 months',
    players: 1243,
    color: Colors.chartRed,
    glow: 'rgba(239, 68, 68, 0.3)',
  },
  {
    id: '2',
    title: 'Dot-com Bubble',
    description: 'Survive the tech bubble burst of 2000',
    difficulty: 'Medium',
    duration: '1 year',
    players: 892,
    color: Colors.gold,
    glow: Colors.glow.orange,
  },
  {
    id: '3',
    title: 'COVID-19 Crash',
    description: 'Trade through the 2020 pandemic volatility',
    difficulty: 'Medium',
    duration: '3 months',
    players: 2156,
    color: Colors.accent,
    glow: Colors.glow.purple,
  },
  {
    id: '4',
    title: 'Bull Market 2017',
    description: 'Maximize gains in a strong bull run',
    difficulty: 'Easy',
    duration: '1 year',
    players: 1567,
    color: Colors.chartGreen,
    glow: Colors.glow.green,
  },
];

export default function GameModeScreen() {
  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      <LinearGradient
        colors={[Colors.backgroundGradientStart, Colors.backgroundGradientEnd]}
        style={styles.gradientBackground}
      >
        <View style={styles.header}>
          <View style={styles.headerIcon}>
            <LinearGradient
              colors={[Colors.gold, Colors.gold]}
              style={styles.iconGradient}
            >
              <Gamepad2 size={32} color={Colors.primary} />
            </LinearGradient>
          </View>
          <Text style={styles.headerTitle}>Game Mode</Text>
          <Text style={styles.headerSubtitle}>
            Test your skills in historical market scenarios
          </Text>
        </View>

        <View style={styles.statsRow}>
          <View style={styles.statCard}>
            <LinearGradient
              colors={[Colors.glow.orange, 'transparent']}
              style={styles.statGradient}
            >
              <View style={styles.statIconBg}>
                <Trophy size={24} color={Colors.gold} />
              </View>
              <Text style={styles.statValue}>12</Text>
              <Text style={styles.statLabel}>Wins</Text>
            </LinearGradient>
          </View>
          <View style={styles.statCard}>
            <LinearGradient
              colors={[Colors.glow.green, 'transparent']}
              style={styles.statGradient}
            >
              <View style={styles.statIconBg}>
                <TrendingUp size={24} color={Colors.chartGreen} />
              </View>
              <Text style={styles.statValue}>+24%</Text>
              <Text style={styles.statLabel}>Avg Return</Text>
            </LinearGradient>
          </View>
          <View style={styles.statCard}>
            <LinearGradient
              colors={[Colors.glow.blue, 'transparent']}
              style={styles.statGradient}
            >
              <View style={styles.statIconBg}>
                <Star size={24} color={Colors.secondary} fill={Colors.secondary} />
              </View>
              <Text style={styles.statValue}>#156</Text>
              <Text style={styles.statLabel}>Rank</Text>
            </LinearGradient>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Historical Scenarios</Text>
          {SCENARIOS.map((scenario) => (
            <View key={scenario.id} style={styles.scenarioCard}>
              <LinearGradient
                colors={[scenario.glow, 'transparent']}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.scenarioGradient}
              >
                <View style={styles.scenarioHeader}>
                  <Text style={styles.scenarioTitle}>{scenario.title}</Text>
                  <View style={[
                    styles.difficultyBadge,
                    {
                      backgroundColor:
                        scenario.difficulty === 'Easy'
                          ? Colors.glow.green
                          : scenario.difficulty === 'Medium'
                          ? Colors.glow.orange
                          : 'rgba(239, 68, 68, 0.2)',
                    },
                  ]}>
                    <Text style={[
                      styles.difficultyText,
                      {
                        color:
                          scenario.difficulty === 'Easy'
                            ? Colors.chartGreen
                            : scenario.difficulty === 'Medium'
                            ? Colors.gold
                            : Colors.chartRed,
                      },
                    ]}>
                      {scenario.difficulty}
                    </Text>
                  </View>
                </View>
                <Text style={styles.scenarioDescription}>{scenario.description}</Text>
                <View style={styles.scenarioMeta}>
                  <View style={styles.metaItem}>
                    <Clock size={16} color={Colors.textMuted} />
                    <Text style={styles.metaText}>{scenario.duration}</Text>
                  </View>
                  <View style={styles.metaItem}>
                    <Users size={16} color={Colors.textMuted} />
                    <Text style={styles.metaText}>{scenario.players} players</Text>
                  </View>
                </View>
                <TouchableOpacity style={[styles.playButton, { backgroundColor: scenario.color }]}>
                  <Play size={20} color={Colors.text} fill={Colors.text} />
                  <Text style={styles.playButtonText}>Start Scenario</Text>
                </TouchableOpacity>
              </LinearGradient>
            </View>
          ))}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Leaderboard</Text>
          <View style={styles.leaderboardCard}>
            {[1, 2, 3, 4, 5].map((rank) => (
              <View key={rank} style={styles.leaderboardRow}>
                <View style={[
                  styles.rankBadge,
                  rank <= 3 && styles.rankBadgeTop,
                ]}>
                  {rank <= 3 ? (
                    <LinearGradient
                      colors={[Colors.gold, Colors.gold]}
                      style={styles.rankBadgeGradient}
                    >
                      <Text style={styles.rankTextTop}>#{rank}</Text>
                    </LinearGradient>
                  ) : (
                    <Text style={styles.rankText}>#{rank}</Text>
                  )}
                </View>
                <View style={styles.playerInfo}>
                  <Text style={styles.playerName}>Player {rank}</Text>
                  <View style={styles.playerScoreRow}>
                    <Zap size={14} color={Colors.gold} fill={Colors.gold} />
                    <Text style={styles.playerScore}>+{(50 - rank * 5).toFixed(1)}% return</Text>
                  </View>
                </View>
                {rank <= 3 && (
                  <View style={styles.trophyBadge}>
                    <Trophy size={16} color={Colors.gold} fill={Colors.gold} />
                  </View>
                )}
              </View>
            ))}
          </View>
        </View>

        <View style={{ height: 40 }} />
      </LinearGradient>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  gradientBackground: {
    flex: 1,
  },
  header: {
    padding: 24,
    paddingTop: 8,
    alignItems: 'center',
  },
  headerIcon: {
    width: 72,
    height: 72,
    borderRadius: 36,
    marginBottom: 16,
    overflow: 'hidden',
  },
  iconGradient: {
    width: 72,
    height: 72,
    borderRadius: 36,
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: '700' as const,
    color: Colors.text,
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 14,
    color: Colors.textSecondary,
    textAlign: 'center',
  },
  statsRow: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    gap: 12,
    marginBottom: 32,
  },
  statCard: {
    flex: 1,
    borderRadius: 20,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: Colors.border,
  },
  statGradient: {
    padding: 20,
    alignItems: 'center',
    gap: 8,
  },
  statIconBg: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: Colors.surfaceLight,
    alignItems: 'center',
    justifyContent: 'center',
  },
  statValue: {
    fontSize: 20,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  statLabel: {
    fontSize: 12,
    color: Colors.textSecondary,
  },
  section: {
    paddingHorizontal: 20,
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '700' as const,
    color: Colors.text,
    marginBottom: 16,
  },
  scenarioCard: {
    borderRadius: 20,
    marginBottom: 16,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: Colors.border,
  },
  scenarioGradient: {
    padding: 20,
    gap: 12,
  },
  scenarioHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  scenarioTitle: {
    fontSize: 18,
    fontWeight: '600' as const,
    color: Colors.text,
    flex: 1,
  },
  difficultyBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  difficultyText: {
    fontSize: 12,
    fontWeight: '600' as const,
  },
  scenarioDescription: {
    fontSize: 14,
    color: Colors.textSecondary,
    lineHeight: 20,
  },
  scenarioMeta: {
    flexDirection: 'row',
    gap: 16,
  },
  metaItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  metaText: {
    fontSize: 12,
    color: Colors.textMuted,
  },
  playButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 14,
    borderRadius: 16,
    marginTop: 8,
  },
  playButtonText: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  leaderboardCard: {
    backgroundColor: Colors.surface,
    padding: 20,
    borderRadius: 20,
    gap: 16,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  leaderboardRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  rankBadge: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: Colors.surfaceLight,
    alignItems: 'center',
    justifyContent: 'center',
  },
  rankBadgeTop: {
    backgroundColor: 'transparent',
    overflow: 'hidden',
  },
  rankBadgeGradient: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
  },
  rankText: {
    fontSize: 14,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  rankTextTop: {
    fontSize: 14,
    fontWeight: '700' as const,
    color: Colors.primary,
  },
  playerInfo: {
    flex: 1,
    gap: 4,
  },
  playerName: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  playerScoreRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  playerScore: {
    fontSize: 12,
    color: Colors.textSecondary,
  },
  trophyBadge: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: Colors.glow.orange,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
