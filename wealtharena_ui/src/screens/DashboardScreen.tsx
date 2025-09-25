import React, { useState, useEffect } from 'react'
import { ScrollView, View, Text, StyleSheet, useColorScheme, TouchableOpacity, TextInput } from 'react-native'
import AsyncStorage from '@react-native-async-storage/async-storage'
import Icon from 'react-native-vector-icons/MaterialCommunityIcons'
import FeatureUnlockSystem from '../components/FeatureUnlockSystem'
import PortfolioSeedGenerator from '../components/PortfolioSeedGenerator'
import { colors } from '../theme/colors'

interface UserProfile {
  experience: 'beginner' | 'intermediate' | 'advanced';
  riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  investmentGoals: string[];
  timeHorizon: 'short' | 'medium' | 'long';
  suggestedPortfolio: string[];
}

interface DashboardScreenProps {
  onNavigateToPortfolioBuilder?: () => void;
  onNavigateToStrategyLab?: () => void;
  onNavigateToGameMode?: () => void;
  onNavigateToTradeSimulator?: () => void;
  onNavigateToAnalytics?: () => void;
  onNavigateToNotifications?: () => void;
}

const DashboardScreen: React.FC<DashboardScreenProps> = ({ 
  onNavigateToPortfolioBuilder,
  onNavigateToStrategyLab,
  onNavigateToGameMode,
  onNavigateToTradeSimulator,
  onNavigateToAnalytics,
  onNavigateToNotifications
}) => {
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [showFeatures, setShowFeatures] = useState(false);
  const [showPortfolio, setShowPortfolio] = useState(false);
  const colorScheme = useColorScheme()
  const isDarkMode = true // Force dark mode
  const c = colors[isDarkMode ? 'dark' : 'light']

  useEffect(() => {
    loadUserProfile();
  }, []);

  const loadUserProfile = async () => {
    try {
      const savedProfile = await AsyncStorage.getItem('userProfile');
      if (savedProfile) {
        setUserProfile(JSON.parse(savedProfile));
      }
    } catch (error) {
      console.error('Error loading user profile:', error);
    }
  };

  if (showFeatures && userProfile) {
    return <FeatureUnlockSystem userProfile={userProfile} onFeaturePress={() => {}} />;
  }

  if (showPortfolio && userProfile) {
    return <PortfolioSeedGenerator userProfile={userProfile} onPortfolioCreate={() => {}} />;
  }

  return (
    <View style={[styles.container, { backgroundColor: c.background }]}>
      {/* Left Sidebar */}
      <View style={[styles.sidebar, { backgroundColor: c.background }]}>
        {/* Logo */}
        <View style={styles.logoContainer}>
          <View style={[styles.logoIcon, { backgroundColor: c.primary }]}>
            <Icon name="trending-up" size={20} color={c.background} />
          </View>
          <Text style={[styles.logoText, { color: c.text }]}>WealthArena</Text>
        </View>

        {/* Search Bar */}
        <View style={[styles.searchContainer, { backgroundColor: c.background, borderColor: c.border }]}>
          <Icon name="magnify" size={16} color={c.textMuted} />
          <TextInput 
            placeholder="Search" 
            placeholderTextColor={c.textMuted}
            style={[styles.searchInput, { color: c.text }]}
          />
        </View>

        {/* Navigation Menu */}
        <View style={styles.navMenu}>
          <TouchableOpacity style={[styles.navItem, { backgroundColor: c.primary }]}>
            <Icon name="home" size={20} color={c.background} />
            <Text style={[styles.navText, { color: c.background }]}>Dashboard</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.navItem} onPress={() => setShowPortfolio(true)}>
            <Icon name="briefcase" size={20} color={c.text} />
            <Text style={[styles.navText, { color: c.text }]}>Portfolio</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.navItem}>
            <Icon name="flask" size={20} color={c.text} />
            <Text style={[styles.navText, { color: c.text }]}>Strategy Lab</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.navItem}>
            <Icon name="triangle" size={20} color={c.text} />
            <Text style={[styles.navText, { color: c.text }]}>Prisme</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.navItem}>
            <Icon name="check-circle" size={20} color={c.text} />
            <Text style={[styles.navText, { color: c.text }]}>Game Mode</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.navItem}>
            <Icon name="handshake" size={20} color={c.text} />
            <Text style={[styles.navText, { color: c.text }]}>Trade Simulator</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.navItem}>
            <Icon name="chart-line" size={20} color={c.text} />
            <Text style={[styles.navText, { color: c.text }]}>Analytics</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.navItem}>
            <Icon name="chart-bar" size={20} color={c.text} />
            <Text style={[styles.navText, { color: c.text }]}>Analytics</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.navItem} onPress={() => setShowFeatures(true)}>
            <Icon name="star" size={20} color={c.text} />
            <Text style={[styles.navText, { color: c.text }]}>Features</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.navItem}>
            <Icon name="cog" size={20} color={c.text} />
            <Text style={[styles.navText, { color: c.text }]}>Settings</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Main Content */}
      <View style={styles.mainContent}>
        {/* Top Header */}
        <View style={[styles.topHeader, { backgroundColor: c.background }]}>
          <View style={styles.headerRight}>
            <TouchableOpacity style={styles.headerIcon}>
              <Icon name="bell" size={20} color={c.text} />
            </TouchableOpacity>
            <TouchableOpacity style={styles.headerIcon}>
              <Icon name="file-document" size={20} color={c.text} />
            </TouchableOpacity>
            <TouchableOpacity style={styles.headerIcon}>
              <Icon name="file-document" size={20} color={c.text} />
              <View style={[styles.badge, { backgroundColor: c.primary }]}>
                <Text style={[styles.badgeText, { color: c.background }]}>7</Text>
              </View>
            </TouchableOpacity>
            <TouchableOpacity style={styles.profileContainer}>
              <View style={[styles.profilePic, { backgroundColor: c.primary }]} />
              <Icon name="chevron-down" size={16} color={c.text} />
            </TouchableOpacity>
          </View>
        </View>

        {/* Dashboard Content */}
        <ScrollView style={styles.dashboardContent} showsVerticalScrollIndicator={false}>
          {/* User Profile Section */}
          {userProfile && (
            <View style={[styles.profileSection, { backgroundColor: c.surface }]}>
              <View style={styles.profileHeader}>
                <View style={[styles.profileIcon, { backgroundColor: c.primary }]}>
                  <Icon name="account" size={24} color={c.background} />
                </View>
                <View style={styles.profileInfo}>
                  <Text style={[styles.profileTitle, { color: c.text }]}>Welcome Back!</Text>
                  <Text style={[styles.profileSubtitle, { color: c.textMuted }]}>
                    {userProfile.experience.charAt(0).toUpperCase() + userProfile.experience.slice(1)} Level Investor
                  </Text>
                </View>
                <TouchableOpacity 
                  style={[styles.profileButton, { backgroundColor: c.primary }]}
                  onPress={() => setShowPortfolio(true)}
                >
                  <Text style={[styles.profileButtonText, { color: c.background }]}>View Portfolio</Text>
                </TouchableOpacity>
              </View>
              
              <View style={styles.profileStats}>
                <View style={styles.statItem}>
                  <Text style={[styles.statLabel, { color: c.textMuted }]}>Risk Tolerance</Text>
                  <Text style={[styles.statValue, { color: c.text }]}>
                    {userProfile.riskTolerance.charAt(0).toUpperCase() + userProfile.riskTolerance.slice(1)}
                  </Text>
                </View>
                <View style={styles.statItem}>
                  <Text style={[styles.statLabel, { color: c.textMuted }]}>Time Horizon</Text>
                  <Text style={[styles.statValue, { color: c.text }]}>
                    {userProfile.timeHorizon.charAt(0).toUpperCase() + userProfile.timeHorizon.slice(1)} Term
                  </Text>
                </View>
                <View style={styles.statItem}>
                  <Text style={[styles.statLabel, { color: c.textMuted }]}>Goals</Text>
                  <Text style={[styles.statValue, { color: c.text }]}>
                    {userProfile.investmentGoals.length} Active
                  </Text>
                </View>
              </View>
            </View>
          )}

          <View style={styles.dashboardGrid}>
            {/* Portfolio Snapshot */}
            <View style={[styles.card, { backgroundColor: c.surface }]}>
              <View style={styles.cardHeader}>
                <Text style={[styles.cardTitle, { color: c.text }]}>Portfolio Snapshot</Text>
                <TouchableOpacity style={[styles.refreshButton, { backgroundColor: c.primary }]}>
                  <Icon name="refresh" size={16} color={c.background} />
                </TouchableOpacity>
              </View>
              
              <View style={styles.portfolioOverview}>
                <View style={styles.portfolioValue}>
                  <Text style={[styles.portfolioAmount, { color: c.text }]}>$24,567.89</Text>
                  <View style={styles.portfolioChange}>
                    <Icon name="trending-up" size={16} color={c.success} />
                    <Text style={[styles.changeText, { color: c.success }]}>+$1,234.56 (+5.3%)</Text>
                  </View>
                </View>
                
                <View style={styles.portfolioChart}>
                  <View style={[styles.chartPlaceholder, { backgroundColor: c.background }]}>
                    <Icon name="chart-line" size={32} color={c.textMuted} />
                    <Text style={[styles.chartLabel, { color: c.textMuted }]}>Portfolio Performance</Text>
                  </View>
                </View>
              </View>
              
              <View style={styles.portfolioAllocations}>
                <Text style={[styles.allocationsTitle, { color: c.text }]}>Asset Allocation</Text>
                <View style={styles.allocationList}>
                  <View style={styles.allocationItem}>
                    <View style={[styles.allocationDot, { backgroundColor: c.primary }]} />
                    <Text style={[styles.allocationLabel, { color: c.text }]}>Stocks</Text>
                    <Text style={[styles.allocationValue, { color: c.text }]}>65%</Text>
                  </View>
                  <View style={styles.allocationItem}>
                    <View style={[styles.allocationDot, { backgroundColor: c.success }]} />
                    <Text style={[styles.allocationLabel, { color: c.text }]}>Bonds</Text>
                    <Text style={[styles.allocationValue, { color: c.text }]}>25%</Text>
                  </View>
                  <View style={styles.allocationItem}>
                    <View style={[styles.allocationDot, { backgroundColor: c.warning }]} />
                    <Text style={[styles.allocationLabel, { color: c.text }]}>Cash</Text>
                    <Text style={[styles.allocationValue, { color: c.text }]}>10%</Text>
                  </View>
                </View>
              </View>
            </View>

            {/* Market Data Feed */}
            <View style={[styles.card, { backgroundColor: c.surface }]}>
              <View style={styles.cardHeader}>
                <Text style={[styles.cardTitle, { color: c.text }]}>Market Data Feed</Text>
                <View style={styles.cardControls}>
                  <TouchableOpacity style={[styles.controlButton, { backgroundColor: c.background }]}>
                    <Icon name="refresh" size={16} color={c.text} />
                  </TouchableOpacity>
                  <TouchableOpacity style={[styles.controlButton, { backgroundColor: c.background }]}>
                    <Icon name="plus" size={16} color={c.text} />
                  </TouchableOpacity>
                  <TouchableOpacity style={[styles.filterButton, { backgroundColor: c.primary }]}>
                    <Text style={[styles.filterText, { color: c.background }]}>All Markets</Text>
                    <Icon name="chevron-down" size={12} color={c.background} />
                  </TouchableOpacity>
                </View>
              </View>
              
              <View style={styles.marketIndices}>
                <View style={styles.indexItem}>
                  <Text style={[styles.indexName, { color: c.text }]}>S&P 500</Text>
                  <Text style={[styles.indexValue, { color: c.text }]}>4,567.89</Text>
                  <View style={styles.indexChange}>
                    <Icon name="trending-up" size={12} color={c.success} />
                    <Text style={[styles.changeText, { color: c.success }]}>+1.2%</Text>
                  </View>
                </View>
                <View style={styles.indexItem}>
                  <Text style={[styles.indexName, { color: c.text }]}>NASDAQ</Text>
                  <Text style={[styles.indexValue, { color: c.text }]}>14,234.56</Text>
                  <View style={styles.indexChange}>
                    <Icon name="trending-down" size={12} color={c.danger} />
                    <Text style={[styles.changeText, { color: c.danger }]}>-0.8%</Text>
                  </View>
                </View>
                <View style={styles.indexItem}>
                  <Text style={[styles.indexName, { color: c.text }]}>DOW</Text>
                  <Text style={[styles.indexValue, { color: c.text }]}>34,567.89</Text>
                  <View style={styles.indexChange}>
                    <Icon name="trending-up" size={12} color={c.success} />
                    <Text style={[styles.changeText, { color: c.success }]}>+0.5%</Text>
                  </View>
                </View>
              </View>
              
              <View style={styles.marketChart}>
                <View style={[styles.chartPlaceholder, { backgroundColor: c.background }]}>
                  <Icon name="chart-candlestick" size={32} color={c.textMuted} />
                  <Text style={[styles.chartLabel, { color: c.textMuted }]}>Market Overview</Text>
                </View>
              </View>
              
              <View style={styles.newsFeed}>
                <Text style={[styles.newsTitle, { color: c.text }]}>Market News</Text>
                <View style={styles.newsItem}>
                  <View style={[styles.newsAvatar, { backgroundColor: c.primary }]} />
                  <View style={styles.newsContent}>
                    <Text style={[styles.newsText, { color: c.text }]}>Fed Signals Rate Cut</Text>
                    <Text style={[styles.newsSubtext, { color: c.textMuted }]}>Federal Reserve hints at potential rate cuts in Q2</Text>
                  </View>
                  <View style={styles.newsRating}>
                    <Icon name="star" size={12} color={c.primary} />
                    <Icon name="star" size={12} color={c.primary} />
                    <Icon name="star" size={12} color={c.primary} />
                    <Icon name="star" size={12} color={c.primary} />
                    <Icon name="star" size={12} color={c.textMuted} />
                  </View>
                </View>
                <View style={styles.newsItem}>
                  <View style={[styles.newsAvatar, { backgroundColor: c.success }]} />
                  <View style={styles.newsContent}>
                    <Text style={[styles.newsText, { color: c.text }]}>Tech Earnings Beat</Text>
                    <Text style={[styles.newsSubtext, { color: c.textMuted }]}>Major tech companies report strong Q4 results</Text>
                  </View>
                  <View style={styles.newsRating}>
                    <Icon name="star" size={12} color={c.primary} />
                    <Icon name="star" size={12} color={c.primary} />
                    <Icon name="star" size={12} color={c.primary} />
                    <Icon name="star" size={12} color={c.primary} />
                    <Icon name="star" size={12} color={c.primary} />
                  </View>
                </View>
              </View>
            </View>

            {/* Progression Tracker */}
            <View style={[styles.card, { backgroundColor: c.background }]}>
              <Text style={[styles.cardTitle, { color: c.text }]}>Progression Tracker</Text>
              <View style={styles.levelContainer}>
                <Text style={[styles.levelText, { color: c.text }]}>Intermediate</Text>
                <Text style={[styles.levelNumber, { color: c.text }]}>5-6</Text>
                <Icon name="diamond" size={16} color={c.text} />
              </View>
              <View style={styles.badgesGrid}>
                <View style={[styles.badge, { backgroundColor: c.textMuted }]} />
                <View style={[styles.badge, { backgroundColor: c.primary }]} />
                <View style={[styles.badge, { backgroundColor: c.textMuted }]} />
                <View style={[styles.badge, { backgroundColor: c.primary }]} />
                <View style={[styles.badge, { backgroundColor: c.textSecondary }]} />
                <View style={[styles.badge, { backgroundColor: c.border }]} />
              </View>
              <View style={styles.progressContainer}>
                <View style={styles.progressItem}>
                  <Text style={[styles.progressLabel, { color: c.text }]}>Ezrred</Text>
                  <View style={[styles.progressBar, { backgroundColor: c.background }]}>
                    <View style={[styles.progressFill, { backgroundColor: c.primary, width: '90%' }]} />
                  </View>
                  <Text style={[styles.progressValue, { color: c.text }]}>5-6</Text>
                </View>
                <View style={styles.progressItem}>
                  <Text style={[styles.progressLabel, { color: c.text }]}>Ligred</Text>
                  <View style={[styles.progressBar, { backgroundColor: c.background }]}>
                    <View style={[styles.progressFill, { backgroundColor: c.primary, width: '85%' }]} />
                  </View>
                  <Icon name="chevron-down" size={16} color={c.text} />
                </View>
              </View>
            </View>

            {/* Quick Actions */}
            <View style={[styles.card, { backgroundColor: c.background }]}>
              <View style={styles.cardHeader}>
                <Text style={[styles.cardTitle, { color: c.text }]}>Quick Actions</Text>
                <TouchableOpacity style={[styles.dropdown, { backgroundColor: c.background, borderColor: c.border }]}>
                  <Text style={[styles.dropdownText, { color: c.text }]}>Termar</Text>
                  <Icon name="chevron-down" size={12} color={c.text} />
                </TouchableOpacity>
              </View>
              <View style={styles.quickActionsGrid}>
                <TouchableOpacity 
                  style={[styles.actionButton, { backgroundColor: c.primary }]}
                  onPress={onNavigateToPortfolioBuilder}
                >
                  <Text style={[styles.actionText, { color: c.background }]}>Portfolio Builder</Text>
                </TouchableOpacity>
                <TouchableOpacity 
                  style={[styles.actionButton, { backgroundColor: c.primary }]}
                  onPress={onNavigateToStrategyLab}
                >
                  <Text style={[styles.actionText, { color: c.background }]}>Strategy Lab</Text>
                </TouchableOpacity>
                <TouchableOpacity 
                  style={[styles.actionButton, { backgroundColor: c.primary }]}
                  onPress={onNavigateToGameMode}
                >
                  <Text style={[styles.actionText, { color: c.background }]}>Historical Game</Text>
                  <Icon name="chevron-right" size={16} color={c.background} />
                </TouchableOpacity>
                <TouchableOpacity 
                  style={[styles.actionButton, { backgroundColor: c.primary }]}
                  onPress={onNavigateToTradeSimulator}
                >
                  <Text style={[styles.actionText, { color: c.background }]}>Trade Simulator</Text>
                </TouchableOpacity>
                <TouchableOpacity 
                  style={[styles.actionButton, { backgroundColor: c.primary }]}
                  onPress={onNavigateToAnalytics}
                >
                  <Text style={[styles.actionText, { color: c.background }]}>Analytics</Text>
                </TouchableOpacity>
                <TouchableOpacity 
                  style={[styles.actionButton, { backgroundColor: c.primary }]}
                  onPress={onNavigateToNotifications}
                >
                  <Text style={[styles.actionText, { color: c.background }]}>Notifications</Text>
                  <Icon name="chevron-right" size={16} color={c.background} />
                </TouchableOpacity>
              </View>
            </View>

            {/* Challenges */}
            <View style={[styles.card, { backgroundColor: c.background }]}>
              <Text style={[styles.cardTitle, { color: c.text }]}>Challenges</Text>
              <View style={styles.challengesList}>
                <View style={styles.challengeItem}>
                  <Icon name="music" size={20} color={c.text} />
                  <Text style={[styles.challengeText, { color: c.text }]}>Completen Ahallenges</Text>
                  <Text style={[styles.challengeValue, { color: c.text }]}>20.04</Text>
                </View>
                <View style={styles.challengeItem}>
                  <Icon name="check-circle" size={20} color={c.primary} />
                  <Text style={[styles.challengeText, { color: c.text }]}>Prompetal Challenges</Text>
                  <Text style={[styles.challengeValue, { color: c.text }]}>29.75</Text>
                </View>
                <View style={styles.challengeItem}>
                  <Icon name="check" size={20} color={c.primary} />
                  <Text style={[styles.challengeText, { color: c.text }]}>Progretan Challenges</Text>
                  <Text style={[styles.challengeValue, { color: c.text }]}>10:0%</Text>
                </View>
              </View>
            </View>
        </View>
      </ScrollView>
      </View>
    </View>
  )
}

export default DashboardScreen

const styles = StyleSheet.create({
  container: { 
    flex: 1, 
    flexDirection: 'row' 
  },
  
  // Sidebar
  sidebar: {
    width: 250,
    padding: 20,
    borderRightWidth: 1,
    borderRightColor: '#333',
  },
  logoContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  logoIcon: {
    width: 32,
    height: 32,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  logoText: {
    fontSize: 18,
    fontWeight: '700',
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    marginBottom: 20,
  },
  searchInput: {
    flex: 1,
    marginLeft: 8,
    fontSize: 14,
  },
  navMenu: {
    gap: 8,
  },
  navItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderRadius: 8,
    marginBottom: 4,
  },
  navText: {
    marginLeft: 12,
    fontSize: 14,
    fontWeight: '500',
  },
  
  // Main Content
  mainContent: {
    flex: 1,
  },
  topHeader: {
    height: 60,
    flexDirection: 'row',
    justifyContent: 'flex-end',
    alignItems: 'center',
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  headerIcon: {
    position: 'relative',
  },
  profileContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  profilePic: {
    width: 32,
    height: 32,
    borderRadius: 16,
  },
  badge: {
    position: 'absolute',
    top: -8,
    right: -8,
    width: 16,
    height: 16,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  badgeText: {
    color: '#FFF',
    fontSize: 10,
    fontWeight: '700',
  },
  
  // Dashboard Content
  dashboardContent: {
    flex: 1,
    padding: 20,
  },
  dashboardGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
  },
  card: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    minHeight: 200,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 16,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  cardControls: {
    flexDirection: 'row',
    gap: 8,
  },
  controlButton: {
    padding: 8,
    borderRadius: 6,
  },
  dropdown: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    gap: 4,
  },
  dropdownText: {
    fontSize: 12,
    fontWeight: '600',
  },
  
  // Portfolio Snapshot
  
  // Market Data Feed
  newsFeed: {
    gap: 8,
  },
  newsItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  newsAvatar: {
    width: 24,
    height: 24,
    borderRadius: 12,
  },
  newsContent: {
    flex: 1,
  },
  newsText: {
    fontSize: 12,
    fontWeight: '600',
  },
  newsSubtext: {
    fontSize: 10,
  },
  newsRating: {
    flexDirection: 'row',
    gap: 2,
  },
  
  // Progression Tracker
  levelContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 16,
  },
  levelText: {
    fontSize: 14,
    fontWeight: '600',
  },
  levelNumber: {
    fontSize: 14,
    fontWeight: '700',
  },
  badgesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 16,
  },
  progressContainer: {
    gap: 12,
  },
  progressItem: {
    gap: 4,
  },
  progressLabel: {
    fontSize: 12,
    fontWeight: '600',
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 4,
  },
  progressValue: {
    fontSize: 12,
    fontWeight: '600',
  },
  
  // Quick Actions
  quickActionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  actionButton: {
    flex: 1,
    minWidth: '30%',
    padding: 12,
    borderRadius: 8,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
  },
  actionText: {
    fontSize: 10,
    fontWeight: '600',
    textAlign: 'center',
  },
  
  // Challenges
  challengesList: {
    gap: 12,
  },
  challengeItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  challengeText: {
    flex: 1,
    fontSize: 12,
    fontWeight: '500',
  },
  challengeValue: {
    fontSize: 12,
    fontWeight: '700',
  },
  // Profile section styles
  profileSection: {
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
  },
  profileHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  profileIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  profileInfo: {
    flex: 1,
  },
  profileTitle: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 4,
  },
  profileSubtitle: {
    fontSize: 14,
  },
  profileButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
  },
  profileButtonText: {
    fontSize: 14,
    fontWeight: '600',
  },
  profileStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statItem: {
    alignItems: 'center',
  },
  statLabel: {
    fontSize: 12,
    marginBottom: 4,
  },
  statValue: {
    fontSize: 14,
    fontWeight: '700',
  },
  
  // Enhanced Portfolio Snapshot
  refreshButton: {
    padding: 8,
    borderRadius: 6,
  },
  portfolioOverview: {
    marginBottom: 16,
  },
  portfolioValue: {
    alignItems: 'center',
    marginBottom: 16,
  },
  portfolioAmount: {
    fontSize: 28,
    fontWeight: '700',
    marginBottom: 8,
  },
  portfolioChange: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  changeText: {
    fontSize: 14,
    fontWeight: '600',
  },
  portfolioChart: {
    height: 100,
    marginBottom: 16,
  },
  chartPlaceholder: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 8,
  },
  chartLabel: {
    fontSize: 12,
    marginTop: 8,
  },
  portfolioAllocations: {
    gap: 12,
  },
  allocationsTitle: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 8,
  },
  allocationList: {
    gap: 8,
  },
  allocationItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  allocationDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  allocationLabel: {
    flex: 1,
    fontSize: 14,
    fontWeight: '500',
  },
  allocationValue: {
    fontSize: 14,
    fontWeight: '700',
  },
  
  // Enhanced Market Data Feed
  filterButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    gap: 4,
  },
  filterText: {
    fontSize: 12,
    fontWeight: '600',
  },
  marketIndices: {
    gap: 12,
    marginBottom: 16,
  },
  indexItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  indexName: {
    fontSize: 14,
    fontWeight: '600',
    flex: 1,
  },
  indexValue: {
    fontSize: 14,
    fontWeight: '700',
    marginRight: 12,
  },
  indexChange: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  marketChart: {
    height: 100,
    marginBottom: 16,
  },
  newsTitle: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 12,
  },
})


