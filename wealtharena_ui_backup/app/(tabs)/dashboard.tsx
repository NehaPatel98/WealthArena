import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import { useRouter } from 'expo-router';
import {
  TrendingUp,
  TrendingDown,
  Briefcase,
  Target,
  Gamepad2,
  BarChart3,
  Bell,
  Award,
  ArrowRight,
  Activity,
  Zap,
  DollarSign,
  ArrowUpRight,
  ArrowDownRight,
} from 'lucide-react-native';
import { useUserTier } from '@/contexts/UserTierContext';
import Colors from '@/constants/colors';
import { LinearGradient } from 'expo-linear-gradient';

const { width } = Dimensions.get('window');

export default function DashboardScreen() {
  const router = useRouter();
  const { profile } = useUserTier();

  const portfolioValue = 125430.50;
  const weeklyChange = 5300;
  const weeklyChangePercent = -1.32;
  const todayChange = 2400;
  const todayChangePercent = -3.44;
  const isWeeklyPositive = weeklyChange >= 0;
  const isTodayPositive = todayChange >= 0;

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      <LinearGradient
        colors={[Colors.backgroundGradientStart, Colors.backgroundGradientEnd]}
        style={styles.gradientBackground}
      >
        <View style={styles.header}>
          <View>
            <View style={styles.avatarSmall}>
              <Text style={styles.avatarEmojiSmall}>🦊</Text>
            </View>
          </View>
          <View style={styles.headerCenter}>
            <Text style={styles.userName}>Victor Grey</Text>
            <Text style={styles.welcomeText}>Welcome Back</Text>
          </View>
          <TouchableOpacity 
            style={styles.menuButton}
            onPress={() => router.push('/notifications' as any)}
          >
            <View style={styles.menuIcon}>
              <View style={styles.menuLine} />
              <View style={styles.menuLine} />
              <View style={styles.menuLine} />
            </View>
          </TouchableOpacity>
        </View>

        <TouchableOpacity 
          style={styles.capitalFlowCard}
          activeOpacity={0.9}
        >
          <View style={styles.capitalFlowHeader}>
            <Text style={styles.capitalFlowTitle}>Capital Flow</Text>
            <ArrowRight size={20} color={Colors.text} />
          </View>

          <View style={styles.capitalFlowContent}>
            <View style={styles.flowItem}>
              <View style={styles.flowBadge}>
                <Text style={styles.flowBadgeText}>WEEKLY</Text>
              </View>
              <View style={styles.flowValueContainer}>
                <Text style={styles.flowValue}>
                  {isWeeklyPositive ? '+' : ''}{(weeklyChange / 1000).toFixed(1)}b
                </Text>
                <View style={styles.flowChange}>
                  <Text style={[styles.flowChangeText, { color: isWeeklyPositive ? Colors.chartGreen : Colors.chartRed }]}>
                    {weeklyChangePercent}%
                  </Text>
                  {isWeeklyPositive ? (
                    <ArrowUpRight size={16} color={Colors.chartGreen} />
                  ) : (
                    <ArrowDownRight size={16} color={Colors.chartRed} />
                  )}
                </View>
              </View>
            </View>

            <View style={styles.flowDivider} />

            <View style={styles.flowItem}>
              <View style={[styles.flowBadge, { backgroundColor: Colors.glow.purple }]}>
                <Text style={styles.flowBadgeText}>TODAY</Text>
              </View>
              <View style={styles.flowValueContainer}>
                <Text style={styles.flowValue}>
                  {isTodayPositive ? '+' : ''}{(todayChange / 1000).toFixed(1)}k
                </Text>
                <View style={styles.flowChange}>
                  <Text style={[styles.flowChangeText, { color: isTodayPositive ? Colors.chartGreen : Colors.chartRed }]}>
                    {todayChangePercent}%
                  </Text>
                  {isTodayPositive ? (
                    <ArrowUpRight size={16} color={Colors.chartGreen} />
                  ) : (
                    <ArrowDownRight size={16} color={Colors.chartRed} />
                  )}
                </View>
              </View>
            </View>
          </View>

          <View style={styles.earningProcess}>
            <View style={styles.earningHeader}>
              <Text style={styles.earningTitle}>Earning Process</Text>
              <Text style={styles.earningPercent}>29%</Text>
            </View>
            
            <View style={styles.donutContainer}>
              <View style={styles.donutChart}>
                <View style={styles.donutCenter}>
                  <DollarSign size={32} color={Colors.text} />
                  <Text style={styles.donutValue}>12k</Text>
                  <Text style={styles.donutLabel}>7 Days</Text>
                </View>
              </View>

              <View style={styles.tokensList}>
                <View style={styles.tokenItem}>
                  <View style={[styles.tokenDot, { backgroundColor: Colors.chartYellow }]} />
                  <Text style={styles.tokenLabel}>TRN</Text>
                </View>
                <View style={styles.tokenItem}>
                  <View style={[styles.tokenDot, { backgroundColor: Colors.textMuted }]} />
                  <Text style={styles.tokenLabel}>HTN</Text>
                </View>
                <View style={styles.tokenItem}>
                  <View style={[styles.tokenDot, { backgroundColor: Colors.accent }]} />
                  <Text style={styles.tokenLabel}>OKB</Text>
                </View>
                <View style={styles.tokenItem}>
                  <View style={[styles.tokenDot, { backgroundColor: Colors.chartCyan }]} />
                  <Text style={styles.tokenLabel}>FTX</Text>
                </View>
              </View>
            </View>
          </View>

          <View style={styles.insertMoney}>
            <Text style={styles.insertMoneyTitle}>Insert Your Money</Text>
            <View style={styles.moneyRow}>
              <View style={styles.moneyInput}>
                <Text style={styles.moneyValue}>$5,333.00</Text>
                <View style={styles.sendButton}>
                  <ArrowUpRight size={16} color={Colors.text} />
                </View>
              </View>
              <View style={styles.coinIconLarge}>
                <Text style={styles.coinEmojiLarge}>🪙</Text>
              </View>
              <View style={styles.moneyInput}>
                <Text style={styles.moneyValue}>$5,333.00</Text>
                <View style={styles.receiveButton}>
                  <ArrowDownRight size={16} color={Colors.text} />
                </View>
              </View>
            </View>
          </View>
        </TouchableOpacity>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Quick Actions</Text>
          <View style={styles.actionsGrid}>
            <TouchableOpacity
              style={styles.actionCard}
              onPress={() => router.push('/portfolio-builder' as any)}
            >
              <LinearGradient
                colors={[Colors.glow.blue, 'transparent']}
                style={styles.actionGradient}
              >
                <View style={[styles.actionIcon, { backgroundColor: Colors.glow.blue }]}>
                  <Briefcase size={24} color={Colors.secondary} />
                </View>
                <Text style={styles.actionTitle}>Portfolio Builder</Text>
                <Text style={styles.actionSubtitle}>Create & optimize</Text>
              </LinearGradient>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.actionCard}
              onPress={() => router.push('/(tabs)/game' as any)}
            >
              <LinearGradient
                colors={[Colors.glow.orange, 'transparent']}
                style={styles.actionGradient}
              >
                <View style={[styles.actionIcon, { backgroundColor: Colors.glow.orange }]}>
                  <Gamepad2 size={24} color={Colors.gold} />
                </View>
                <Text style={styles.actionTitle}>Game Mode</Text>
                <Text style={styles.actionSubtitle}>Historical scenarios</Text>
              </LinearGradient>
            </TouchableOpacity>

            {profile.tier === 'intermediate' && (
              <>
                <TouchableOpacity
                  style={styles.actionCard}
                  onPress={() => router.push('/strategy-lab' as any)}
                >
                  <LinearGradient
                    colors={[Colors.glow.purple, 'transparent']}
                    style={styles.actionGradient}
                  >
                    <View style={[styles.actionIcon, { backgroundColor: Colors.glow.purple }]}>
                      <Target size={24} color={Colors.accent} />
                    </View>
                    <Text style={styles.actionTitle}>Strategy Lab</Text>
                    <Text style={styles.actionSubtitle}>Build strategies</Text>
                  </LinearGradient>
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.actionCard}
                  onPress={() => router.push('/analytics' as any)}
                >
                  <LinearGradient
                    colors={[Colors.glow.cyan, 'transparent']}
                    style={styles.actionGradient}
                  >
                    <View style={[styles.actionIcon, { backgroundColor: Colors.glow.cyan }]}>
                      <BarChart3 size={24} color={Colors.chartCyan} />
                    </View>
                    <Text style={styles.actionTitle}>Analytics</Text>
                    <Text style={styles.actionSubtitle}>Deep insights</Text>
                  </LinearGradient>
                </TouchableOpacity>
              </>
            )}
          </View>
        </View>

        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Progression Tracker</Text>
            <TouchableOpacity onPress={() => router.push('/(tabs)/profile' as any)}>
              <Text style={styles.seeAllText}>View All</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.progressCard}>
            <View style={styles.progressHeader}>
              <View style={styles.tierBadgeSmall}>
                <Award size={16} color={profile.tier === 'beginner' ? Colors.beginner : Colors.intermediate} />
                <Text style={styles.tierTextSmall}>
                  {profile.tier === 'beginner' ? 'Beginner' : 'Intermediate'}
                </Text>
              </View>
              <Text style={styles.progressValue}>
                {profile.completedChallenges}/{profile.totalChallenges}
              </Text>
            </View>
            <View style={styles.progressBar}>
              <LinearGradient
                colors={[Colors.secondary, Colors.accent]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
                style={[
                  styles.progressFill,
                  { width: `${(profile.completedChallenges / profile.totalChallenges) * 100}%` },
                ]}
              />
            </View>
            <View style={styles.achievementsPreview}>
              {profile.achievements.slice(0, 3).map((achievement, index) => (
                <View key={index} style={styles.achievementBadge}>
                  <Award size={14} color={Colors.gold} />
                </View>
              ))}
              {profile.achievements.length > 3 && (
                <Text style={styles.moreAchievements}>+{profile.achievements.length - 3}</Text>
              )}
            </View>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Market Alerts</Text>
          <TouchableOpacity 
            style={styles.alertCard}
            onPress={() => router.push('/notifications' as any)}
          >
            <View style={styles.alertIcon}>
              <Bell size={20} color={Colors.gold} />
            </View>
            <View style={styles.alertContent}>
              <Text style={styles.alertTitle}>Portfolio Rebalancing Suggested</Text>
              <Text style={styles.alertSubtitle}>Your tech allocation is 5% over target</Text>
            </View>
            <ArrowRight size={20} color={Colors.textMuted} />
          </TouchableOpacity>
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
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    paddingTop: 8,
    gap: 12,
  },
  avatarSmall: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: Colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: Colors.secondary,
  },
  avatarEmojiSmall: {
    fontSize: 24,
  },
  headerCenter: {
    flex: 1,
  },
  userName: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  welcomeText: {
    fontSize: 12,
    color: Colors.textSecondary,
  },
  menuButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: Colors.border,
  },
  menuIcon: {
    gap: 4,
  },
  menuLine: {
    width: 16,
    height: 2,
    backgroundColor: Colors.text,
    borderRadius: 1,
  },
  capitalFlowCard: {
    backgroundColor: Colors.surface,
    marginHorizontal: 20,
    padding: 20,
    borderRadius: 24,
    marginBottom: 24,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  capitalFlowHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  capitalFlowTitle: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  capitalFlowContent: {
    flexDirection: 'row',
    marginBottom: 24,
  },
  flowItem: {
    flex: 1,
    gap: 12,
  },
  flowBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    backgroundColor: Colors.glow.blue,
  },
  flowBadgeText: {
    fontSize: 10,
    fontWeight: '700' as const,
    color: Colors.text,
    letterSpacing: 0.5,
  },
  flowValueContainer: {
    gap: 4,
  },
  flowValue: {
    fontSize: 32,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  flowChange: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  flowChangeText: {
    fontSize: 14,
    fontWeight: '600' as const,
  },
  flowDivider: {
    width: 1,
    backgroundColor: Colors.border,
    marginHorizontal: 16,
  },
  earningProcess: {
    marginBottom: 24,
  },
  earningHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  earningTitle: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  earningPercent: {
    fontSize: 24,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  donutContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  donutChart: {
    width: 140,
    height: 140,
    borderRadius: 70,
    borderWidth: 20,
    borderColor: Colors.surfaceLight,
    borderTopColor: Colors.chartYellow,
    borderRightColor: Colors.textMuted,
    borderBottomColor: Colors.chartCyan,
    borderLeftColor: Colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
    transform: [{ rotate: '-45deg' }],
  },
  donutCenter: {
    alignItems: 'center',
    transform: [{ rotate: '45deg' }],
  },
  donutValue: {
    fontSize: 24,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  donutLabel: {
    fontSize: 12,
    color: Colors.textSecondary,
  },
  tokensList: {
    gap: 12,
  },
  tokenItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  tokenDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  tokenLabel: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  insertMoney: {
    gap: 12,
  },
  insertMoneyTitle: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  moneyRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  moneyInput: {
    flex: 1,
    backgroundColor: Colors.surfaceLight,
    borderRadius: 16,
    padding: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    borderWidth: 1,
    borderColor: Colors.border,
  },
  moneyValue: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  sendButton: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: Colors.secondary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  receiveButton: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: Colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
  },
  coinIconLarge: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: Colors.gold,
    alignItems: 'center',
    justifyContent: 'center',
  },
  coinEmojiLarge: {
    fontSize: 24,
  },
  section: {
    paddingHorizontal: 20,
    marginBottom: 24,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '700' as const,
    color: Colors.text,
    marginBottom: 16,
  },
  seeAllText: {
    fontSize: 14,
    color: Colors.secondary,
    fontWeight: '600' as const,
  },
  actionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  actionCard: {
    width: (width - 52) / 2,
    borderRadius: 20,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: Colors.border,
  },
  actionGradient: {
    padding: 20,
    gap: 12,
  },
  actionIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  actionTitle: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  actionSubtitle: {
    fontSize: 12,
    color: Colors.textSecondary,
  },
  progressCard: {
    backgroundColor: Colors.surface,
    padding: 20,
    borderRadius: 20,
    gap: 16,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  progressHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  tierBadgeSmall: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: Colors.surfaceLight,
    borderRadius: 16,
  },
  tierTextSmall: {
    fontSize: 12,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  progressValue: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.secondary,
  },
  progressBar: {
    height: 8,
    backgroundColor: Colors.surfaceLight,
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 4,
  },
  achievementsPreview: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  achievementBadge: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: Colors.glow.orange,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: Colors.gold,
  },
  moreAchievements: {
    fontSize: 12,
    color: Colors.textSecondary,
    fontWeight: '600' as const,
  },
  alertCard: {
    backgroundColor: Colors.surface,
    padding: 16,
    borderRadius: 20,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  alertIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.glow.orange,
    alignItems: 'center',
    justifyContent: 'center',
  },
  alertContent: {
    flex: 1,
    gap: 4,
  },
  alertTitle: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  alertSubtitle: {
    fontSize: 12,
    color: Colors.textSecondary,
  },
});
