import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import {
  User,
  Award,
  Settings,
  Shield,
  Bell,
  HelpCircle,
  LogOut,
  ChevronRight,
  TrendingUp,
  Brain,
  BarChart3,
  Zap,
  Star,
  Trophy,
} from 'lucide-react-native';
import { useUserTier } from '@/contexts/UserTierContext';
import Colors from '@/constants/colors';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';

const { width } = Dimensions.get('window');

const COLLECTIBLES = [
  { id: '1', emoji: '💎', name: 'Diamond', rarity: 'legendary' },
  { id: '2', emoji: '🔮', name: 'Crystal', rarity: 'epic' },
  { id: '3', emoji: '⚡', name: 'Lightning', rarity: 'rare' },
  { id: '4', emoji: '🎯', name: 'Target', rarity: 'common' },
];

const ACHIEVEMENTS = [
  { id: '1', title: 'First Trade', icon: '🎯', unlocked: true },
  { id: '2', title: 'Portfolio Master', icon: '💼', unlocked: true },
  { id: '3', title: 'Risk Manager', icon: '🛡️', unlocked: true },
  { id: '4', title: 'Market Survivor', icon: '🏆', unlocked: false },
  { id: '5', title: 'Bull Run Champion', icon: '🚀', unlocked: false },
  { id: '6', title: 'Strategy Expert', icon: '🎓', unlocked: false },
];

export default function ProfileScreen() {
  const { profile } = useUserTier();
  const router = useRouter();

  const level = 5;
  const currentXP = 445;
  const nextLevelXP = 500;
  const xpProgress = (currentXP / nextLevelXP) * 100;

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      <LinearGradient
        colors={[Colors.backgroundGradientStart, Colors.backgroundGradientEnd]}
        style={styles.gradientBackground}
      >
        <View style={styles.profileHeader}>
          <View style={styles.headerTop}>
            <Text style={styles.headerTitle}>Profile</Text>
            <View style={styles.headerIcons}>
              <TouchableOpacity 
                style={styles.iconButton}
                onPress={() => router.push('/notifications' as any)}
              >
                <Bell size={20} color={Colors.text} />
                <View style={styles.notificationDot} />
              </TouchableOpacity>
              <TouchableOpacity style={styles.iconButton}>
                <Settings size={20} color={Colors.text} />
              </TouchableOpacity>
            </View>
          </View>

          <View style={styles.profileCard}>
            <View style={styles.avatarSection}>
              <View style={styles.avatarContainer}>
                <LinearGradient
                  colors={[Colors.accent, Colors.secondary]}
                  style={styles.avatarGradient}
                >
                  <View style={styles.avatar}>
                    <Text style={styles.avatarEmoji}>🦊</Text>
                  </View>
                </LinearGradient>
                <View style={styles.levelBadge}>
                  <Text style={styles.levelText}>{level}</Text>
                </View>
              </View>

              <View style={styles.profileInfo}>
                <View style={styles.nameRow}>
                  <Text style={styles.profileName}>Hey 👋</Text>
                  <View style={styles.verifiedBadge}>
                    <Star size={12} color={Colors.gold} fill={Colors.gold} />
                  </View>
                </View>
                <Text style={styles.username}>@{profile.name?.toLowerCase().replace(' ', '') || 'investor'}</Text>
                <View style={styles.earningsRow}>
                  <Text style={styles.earningsLabel}>+445/h</Text>
                  <View style={styles.coinIcon}>
                    <Text style={styles.coinEmoji}>🪙</Text>
                  </View>
                </View>
              </View>
            </View>

            <View style={styles.levelSection}>
              <View style={styles.levelHeader}>
                <Text style={styles.levelLabel}>Level</Text>
                <Text style={styles.levelValue}>{level}/10</Text>
              </View>
              <View style={styles.progressBarContainer}>
                <View style={styles.progressBarBackground}>
                  <LinearGradient
                    colors={[Colors.secondary, Colors.accent]}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 0 }}
                    style={[styles.progressBarFill, { width: `${xpProgress}%` }]}
                  />
                </View>
              </View>
            </View>
          </View>

          <View style={styles.characterCard}>
            <View style={styles.characterContainer}>
              <LinearGradient
                colors={[Colors.accent, Colors.secondary]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.characterBackground}
              >
                <Text style={styles.characterEmoji}>🦁</Text>
                <View style={styles.characterBadge}>
                  <Text style={styles.characterBadgeText}>43%</Text>
                </View>
                <View style={styles.characterXP}>
                  <Zap size={16} color={Colors.gold} fill={Colors.gold} />
                  <Text style={styles.characterXPText}>+8</Text>
                </View>
              </LinearGradient>
            </View>

            <View style={styles.collectiblesRow}>
              {COLLECTIBLES.map((item) => (
                <View key={item.id} style={styles.collectibleItem}>
                  <View style={[
                    styles.collectibleBg,
                    item.rarity === 'legendary' && styles.collectibleLegendary,
                    item.rarity === 'epic' && styles.collectibleEpic,
                    item.rarity === 'rare' && styles.collectibleRare,
                  ]}>
                    <Text style={styles.collectibleEmoji}>{item.emoji}</Text>
                  </View>
                </View>
              ))}
            </View>

            <View style={styles.navButtons}>
              <TouchableOpacity style={styles.navButton}>
                <View style={styles.navButtonIcon}>
                  <Trophy size={20} color={Colors.text} />
                </View>
              </TouchableOpacity>
              <TouchableOpacity style={styles.navButton}>
                <View style={styles.navButtonIcon}>
                  <BarChart3 size={20} color={Colors.text} />
                </View>
              </TouchableOpacity>
              <TouchableOpacity style={styles.navButton}>
                <View style={styles.navButtonIcon}>
                  <Award size={20} color={Colors.text} />
                </View>
              </TouchableOpacity>
              <TouchableOpacity style={styles.navButton}>
                <View style={styles.navButtonIcon}>
                  <User size={20} color={Colors.text} />
                </View>
              </TouchableOpacity>
            </View>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Achievements</Text>
          <View style={styles.achievementsGrid}>
            {ACHIEVEMENTS.map((achievement) => (
              <View
                key={achievement.id}
                style={[
                  styles.achievementCard,
                  !achievement.unlocked && styles.achievementLocked,
                ]}
              >
                <View style={[
                  styles.achievementIconBg,
                  achievement.unlocked && styles.achievementIconBgUnlocked,
                ]}>
                  <Text style={styles.achievementIcon}>{achievement.icon}</Text>
                </View>
                <Text style={[
                  styles.achievementTitle,
                  !achievement.unlocked && styles.achievementTitleLocked,
                ]}>
                  {achievement.title}
                </Text>
              </View>
            ))}
          </View>
        </View>

        {profile.tier === 'intermediate' && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Advanced Tools</Text>
            <View style={styles.settingsCard}>
              <TouchableOpacity 
                style={styles.settingRow}
                onPress={() => router.push('/trade-simulator' as any)}
              >
                <View style={styles.settingLeft}>
                  <View style={[styles.settingIconBg, { backgroundColor: Colors.glow.orange }]}>
                    <TrendingUp size={20} color={Colors.gold} />
                  </View>
                  <Text style={styles.settingText}>Trade Simulator</Text>
                </View>
                <ChevronRight size={20} color={Colors.textMuted} />
              </TouchableOpacity>

              <View style={styles.settingDivider} />

              <TouchableOpacity 
                style={styles.settingRow}
                onPress={() => router.push('/explainability' as any)}
              >
                <View style={styles.settingLeft}>
                  <View style={[styles.settingIconBg, { backgroundColor: Colors.glow.purple }]}>
                    <Brain size={20} color={Colors.accent} />
                  </View>
                  <Text style={styles.settingText}>Explainability & Audit</Text>
                </View>
                <ChevronRight size={20} color={Colors.textMuted} />
              </TouchableOpacity>

              <View style={styles.settingDivider} />

              <TouchableOpacity 
                style={styles.settingRow}
                onPress={() => router.push('/admin-portal' as any)}
              >
                <View style={styles.settingLeft}>
                  <View style={[styles.settingIconBg, { backgroundColor: Colors.glow.blue }]}>
                    <BarChart3 size={20} color={Colors.secondary} />
                  </View>
                  <Text style={styles.settingText}>Admin Portal</Text>
                </View>
                <ChevronRight size={20} color={Colors.textMuted} />
              </TouchableOpacity>
            </View>
          </View>
        )}

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Settings</Text>
          <View style={styles.settingsCard}>
            <TouchableOpacity style={styles.settingRow}>
              <View style={styles.settingLeft}>
                <View style={[styles.settingIconBg, { backgroundColor: Colors.glow.blue }]}>
                  <Settings size={20} color={Colors.secondary} />
                </View>
                <Text style={styles.settingText}>Account Settings</Text>
              </View>
              <ChevronRight size={20} color={Colors.textMuted} />
            </TouchableOpacity>

            <View style={styles.settingDivider} />

            <TouchableOpacity 
              style={styles.settingRow}
              onPress={() => router.push('/notifications' as any)}
            >
              <View style={styles.settingLeft}>
                <View style={[styles.settingIconBg, { backgroundColor: Colors.glow.purple }]}>
                  <Bell size={20} color={Colors.accent} />
                </View>
                <Text style={styles.settingText}>Notifications</Text>
              </View>
              <ChevronRight size={20} color={Colors.textMuted} />
            </TouchableOpacity>

            <View style={styles.settingDivider} />

            <TouchableOpacity style={styles.settingRow}>
              <View style={styles.settingLeft}>
                <View style={[styles.settingIconBg, { backgroundColor: Colors.glow.green }]}>
                  <Shield size={20} color={Colors.success} />
                </View>
                <Text style={styles.settingText}>Privacy & Security</Text>
              </View>
              <ChevronRight size={20} color={Colors.textMuted} />
            </TouchableOpacity>

            <View style={styles.settingDivider} />

            <TouchableOpacity style={styles.settingRow}>
              <View style={styles.settingLeft}>
                <View style={[styles.settingIconBg, { backgroundColor: Colors.glow.cyan }]}>
                  <HelpCircle size={20} color={Colors.chartCyan} />
                </View>
                <Text style={styles.settingText}>Help & Support</Text>
              </View>
              <ChevronRight size={20} color={Colors.textMuted} />
            </TouchableOpacity>

            <View style={styles.settingDivider} />

            <TouchableOpacity style={styles.settingRow}>
              <View style={styles.settingLeft}>
                <View style={[styles.settingIconBg, { backgroundColor: 'rgba(239, 68, 68, 0.2)' }]}>
                  <LogOut size={20} color={Colors.danger} />
                </View>
                <Text style={[styles.settingText, { color: Colors.danger }]}>Log Out</Text>
              </View>
              <ChevronRight size={20} color={Colors.textMuted} />
            </TouchableOpacity>
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
  profileHeader: {
    padding: 20,
    paddingTop: 8,
  },
  headerTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  headerIcons: {
    flexDirection: 'row',
    gap: 12,
  },
  iconButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: Colors.border,
    position: 'relative',
  },
  notificationDot: {
    position: 'absolute',
    top: 8,
    right: 8,
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: Colors.danger,
  },
  profileCard: {
    backgroundColor: Colors.surface,
    borderRadius: 24,
    padding: 20,
    borderWidth: 1,
    borderColor: Colors.border,
    marginBottom: 16,
  },
  avatarSection: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  avatarContainer: {
    position: 'relative',
    marginRight: 16,
  },
  avatarGradient: {
    width: 80,
    height: 80,
    borderRadius: 40,
    padding: 3,
  },
  avatar: {
    width: 74,
    height: 74,
    borderRadius: 37,
    backgroundColor: Colors.primaryLight,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarEmoji: {
    fontSize: 40,
  },
  levelBadge: {
    position: 'absolute',
    bottom: -4,
    right: -4,
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: Colors.gold,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 3,
    borderColor: Colors.primaryLight,
  },
  levelText: {
    fontSize: 14,
    fontWeight: '700' as const,
    color: Colors.primary,
  },
  profileInfo: {
    flex: 1,
  },
  nameRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 4,
  },
  profileName: {
    fontSize: 20,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  verifiedBadge: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: Colors.glow.orange,
    alignItems: 'center',
    justifyContent: 'center',
  },
  username: {
    fontSize: 14,
    color: Colors.textSecondary,
    marginBottom: 8,
  },
  earningsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  earningsLabel: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.gold,
  },
  coinIcon: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: Colors.glow.orange,
    alignItems: 'center',
    justifyContent: 'center',
  },
  coinEmoji: {
    fontSize: 14,
  },
  levelSection: {
    gap: 8,
  },
  levelHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  levelLabel: {
    fontSize: 14,
    color: Colors.textSecondary,
    fontWeight: '600' as const,
  },
  levelValue: {
    fontSize: 14,
    color: Colors.text,
    fontWeight: '700' as const,
  },
  progressBarContainer: {
    height: 8,
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressBarBackground: {
    flex: 1,
    backgroundColor: Colors.surfaceLight,
    borderRadius: 4,
  },
  progressBarFill: {
    height: '100%',
    borderRadius: 4,
  },
  characterCard: {
    backgroundColor: Colors.surface,
    borderRadius: 24,
    padding: 20,
    borderWidth: 1,
    borderColor: Colors.border,
    gap: 16,
  },
  characterContainer: {
    alignSelf: 'center',
  },
  characterBackground: {
    width: 200,
    height: 200,
    borderRadius: 100,
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
  },
  characterEmoji: {
    fontSize: 100,
  },
  characterBadge: {
    position: 'absolute',
    top: 20,
    right: 20,
    backgroundColor: Colors.primary,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: Colors.text,
  },
  characterBadgeText: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  characterXP: {
    position: 'absolute',
    bottom: 20,
    left: 20,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: Colors.primary,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: Colors.gold,
  },
  characterXPText: {
    fontSize: 14,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  collectiblesRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 12,
  },
  collectibleItem: {
    width: 60,
    height: 60,
  },
  collectibleBg: {
    width: 60,
    height: 60,
    borderRadius: 16,
    backgroundColor: Colors.surfaceLight,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: Colors.border,
  },
  collectibleLegendary: {
    borderColor: Colors.gold,
    backgroundColor: Colors.glow.orange,
  },
  collectibleEpic: {
    borderColor: Colors.accent,
    backgroundColor: Colors.glow.purple,
  },
  collectibleRare: {
    borderColor: Colors.secondary,
    backgroundColor: Colors.glow.blue,
  },
  collectibleEmoji: {
    fontSize: 28,
  },
  navButtons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingTop: 8,
  },
  navButton: {
    width: 56,
    height: 56,
    borderRadius: 28,
    alignItems: 'center',
    justifyContent: 'center',
  },
  navButtonIcon: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: Colors.surfaceLight,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: Colors.border,
  },
  section: {
    paddingHorizontal: 20,
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '700' as const,
    color: Colors.text,
    marginBottom: 16,
  },
  achievementsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  achievementCard: {
    width: (width - 64) / 3,
    backgroundColor: Colors.surface,
    padding: 16,
    borderRadius: 16,
    alignItems: 'center',
    gap: 8,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  achievementLocked: {
    opacity: 0.4,
  },
  achievementIconBg: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: Colors.surfaceLight,
    alignItems: 'center',
    justifyContent: 'center',
  },
  achievementIconBgUnlocked: {
    backgroundColor: Colors.glow.blue,
  },
  achievementIcon: {
    fontSize: 24,
  },
  achievementTitle: {
    fontSize: 11,
    fontWeight: '600' as const,
    color: Colors.text,
    textAlign: 'center',
  },
  achievementTitleLocked: {
    color: Colors.textMuted,
  },
  settingsCard: {
    backgroundColor: Colors.surface,
    borderRadius: 20,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: Colors.border,
  },
  settingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
  },
  settingLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  settingIconBg: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  settingText: {
    fontSize: 16,
    color: Colors.text,
    fontWeight: '500' as const,
  },
  settingDivider: {
    height: 1,
    backgroundColor: Colors.border,
    marginHorizontal: 16,
  },
});
