import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { colors } from '../theme/colors';

interface GameScenario {
  id: string;
  name: string;
  description: string;
  period: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  duration: string;
  startingCapital: number;
  marketConditions: string[];
  challenges: string[];
  leaderboard: LeaderboardEntry[];
}

interface LeaderboardEntry {
  rank: number;
  playerName: string;
  score: number;
  return: number;
  isCurrentUser?: boolean;
}

interface GameModeScreenProps {
  userProfile: {
    experience: 'beginner' | 'intermediate' | 'advanced';
  };
  onBack: () => void;
}

const GameModeScreen: React.FC<GameModeScreenProps> = ({ userProfile, onBack }) => {
  const [activeTab, setActiveTab] = useState<'scenarios' | 'active' | 'leaderboard' | 'replay'>('scenarios');
  const [selectedScenario, setSelectedScenario] = useState<GameScenario | null>(null);
  const [isGameActive, setIsGameActive] = useState(false);
  const [gameProgress, setGameProgress] = useState(0);
  
  const isDarkMode = true;
  const c = colors[isDarkMode ? 'dark' : 'light'];

  const gameScenarios: GameScenario[] = [
    {
      id: '2008-crash',
      name: '2008 Financial Crisis',
      description: 'Navigate through the worst financial crisis since the Great Depression',
      period: '2007-2009',
      difficulty: 'advanced',
      duration: '2 years',
      startingCapital: 100000,
      marketConditions: ['Housing bubble burst', 'Bank failures', 'Credit freeze', 'Massive unemployment'],
      challenges: ['Survive the crash', 'Identify recovery opportunities', 'Beat the market by 20%'],
      leaderboard: [
        { rank: 1, playerName: 'CrisisMaster', score: 95, return: 15.2, isCurrentUser: false },
        { rank: 2, playerName: 'You', score: 87, return: 8.7, isCurrentUser: true },
        { rank: 3, playerName: 'BullTrader', score: 82, return: 5.3, isCurrentUser: false },
      ]
    },
    {
      id: 'dot-com',
      name: 'Dot-com Bubble',
      description: 'Experience the rise and fall of the internet boom',
      period: '1998-2002',
      difficulty: 'intermediate',
      duration: '4 years',
      startingCapital: 50000,
      marketConditions: ['Tech boom', 'IPO frenzy', 'Valuation madness', 'Bubble burst'],
      challenges: ['Ride the tech wave', 'Avoid the crash', 'Find value in the wreckage'],
      leaderboard: [
        { rank: 1, playerName: 'TechGuru', score: 92, return: 45.8, isCurrentUser: false },
        { rank: 2, playerName: 'BubbleBuster', score: 88, return: 32.1, isCurrentUser: false },
        { rank: 3, playerName: 'You', score: 75, return: 18.4, isCurrentUser: true },
      ]
    },
    {
      id: 'covid-19',
      name: 'COVID-19 Market Crash',
      description: 'Navigate the fastest market crash in history',
      period: '2020',
      difficulty: 'beginner',
      duration: '1 year',
      startingCapital: 25000,
      marketConditions: ['Pandemic panic', 'Lockdowns', 'Vaccine race', 'Recovery'],
      challenges: ['Survive the crash', 'Buy the dip', 'Ride the recovery'],
      leaderboard: [
        { rank: 1, playerName: 'You', score: 96, return: 28.5, isCurrentUser: true },
        { rank: 2, playerName: 'VaccineTrader', score: 89, return: 22.1, isCurrentUser: false },
        { rank: 3, playerName: 'RecoveryKing', score: 85, return: 19.7, isCurrentUser: false },
      ]
    }
  ];

  const startGame = (scenario: GameScenario) => {
    setSelectedScenario(scenario);
    setIsGameActive(true);
    setGameProgress(0);
    Alert.alert(
      'Game Started!',
      `You're now playing ${scenario.name}. Good luck!`,
      [{ text: 'OK', onPress: () => setActiveTab('active') }]
    );
  };

  const renderScenarios = () => (
    <View style={styles.tabContent}>
      <Text style={[styles.sectionTitle, { color: c.text }]}>Historical Scenarios</Text>
      <Text style={[styles.sectionDescription, { color: c.textMuted }]}>
        Experience real market events and test your skills
      </Text>
      
      <View style={styles.scenariosGrid}>
        {gameScenarios.map((scenario) => (
          <TouchableOpacity
            key={scenario.id}
            style={[styles.scenarioCard, { backgroundColor: c.surface }]}
            onPress={() => startGame(scenario)}
          >
            <View style={styles.scenarioHeader}>
              <Text style={[styles.scenarioName, { color: c.text }]}>{scenario.name}</Text>
              <View style={[styles.difficultyBadge, { 
                backgroundColor: scenario.difficulty === 'beginner' ? c.success : 
                               scenario.difficulty === 'intermediate' ? c.warning : c.danger 
              }]}>
                <Text style={[styles.difficultyText, { color: c.background }]}>
                  {scenario.difficulty.toUpperCase()}
                </Text>
              </View>
            </View>
            
            <Text style={[styles.scenarioDescription, { color: c.textMuted }]}>
              {scenario.description}
            </Text>
            
            <View style={styles.scenarioInfo}>
              <View style={styles.infoItem}>
                <Icon name="calendar" size={16} color={c.textMuted} />
                <Text style={[styles.infoText, { color: c.textMuted }]}>{scenario.period}</Text>
              </View>
              <View style={styles.infoItem}>
                <Icon name="clock" size={16} color={c.textMuted} />
                <Text style={[styles.infoText, { color: c.textMuted }]}>{scenario.duration}</Text>
              </View>
              <View style={styles.infoItem}>
                <Icon name="currency-usd" size={16} color={c.textMuted} />
                <Text style={[styles.infoText, { color: c.textMuted }]}>
                  ${scenario.startingCapital.toLocaleString()}
                </Text>
              </View>
            </View>
            
            <View style={styles.scenarioChallenges}>
              <Text style={[styles.challengesTitle, { color: c.text }]}>Challenges:</Text>
              {scenario.challenges.map((challenge, index) => (
                <Text key={index} style={[styles.challengeText, { color: c.textMuted }]}>
                  • {challenge}
                </Text>
              ))}
            </View>
            
            <View style={styles.scenarioActions}>
              <TouchableOpacity style={[styles.playButton, { backgroundColor: c.primary }]}>
                <Icon name="play" size={16} color={c.background} />
                <Text style={[styles.playButtonText, { color: c.background }]}>Start Game</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.previewButton, { borderColor: c.border }]}>
                <Icon name="eye" size={16} color={c.text} />
                <Text style={[styles.previewButtonText, { color: c.text }]}>Preview</Text>
              </TouchableOpacity>
            </View>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );

  const renderActiveGame = () => (
    <View style={styles.tabContent}>
      <Text style={[styles.sectionTitle, { color: c.text }]}>Active Game</Text>
      
      {isGameActive && selectedScenario ? (
        <View style={[styles.gameContainer, { backgroundColor: c.surface }]}>
          <View style={styles.gameHeader}>
            <Text style={[styles.gameTitle, { color: c.text }]}>{selectedScenario.name}</Text>
            <View style={styles.gameProgress}>
              <Text style={[styles.progressText, { color: c.textMuted }]}>
                Progress: {gameProgress}%
              </Text>
              <View style={[styles.progressBar, { backgroundColor: c.border }]}>
                <View style={[styles.progressFill, { 
                  backgroundColor: c.primary, 
                  width: `${gameProgress}%` 
                }]} />
              </View>
            </View>
          </View>
          
          <View style={styles.gameControls}>
            <TouchableOpacity style={[styles.controlButton, { backgroundColor: c.success }]}>
              <Icon name="play" size={20} color={c.background} />
              <Text style={[styles.controlText, { color: c.background }]}>Play</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.controlButton, { backgroundColor: c.warning }]}>
              <Icon name="pause" size={20} color={c.background} />
              <Text style={[styles.controlText, { color: c.background }]}>Pause</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.controlButton, { backgroundColor: c.danger }]}>
              <Icon name="stop" size={20} color={c.background} />
              <Text style={[styles.controlText, { color: c.background }]}>Stop</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.controlButton, { backgroundColor: c.primary }]}>
              <Icon name="rewind" size={20} color={c.background} />
              <Text style={[styles.controlText, { color: c.background }]}>Rewind</Text>
            </TouchableOpacity>
          </View>
          
          <View style={styles.gameStats}>
            <View style={styles.statItem}>
              <Text style={[styles.statLabel, { color: c.textMuted }]}>Current Value</Text>
              <Text style={[styles.statValue, { color: c.text }]}>$127,450</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={[styles.statLabel, { color: c.textMuted }]}>P&L</Text>
              <Text style={[styles.statValue, { color: c.success }]}>+$27,450</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={[styles.statLabel, { color: c.textMuted }]}>Return</Text>
              <Text style={[styles.statValue, { color: c.success }]}>+27.45%</Text>
            </View>
          </View>
        </View>
      ) : (
        <View style={styles.noActiveGame}>
          <Icon name="gamepad-variant" size={48} color={c.textMuted} />
          <Text style={[styles.noGameText, { color: c.text }]}>No Active Game</Text>
          <Text style={[styles.noGameSubtext, { color: c.textMuted }]}>
            Select a scenario to start playing
          </Text>
        </View>
      )}
    </View>
  );

  const renderLeaderboard = () => (
    <View style={styles.tabContent}>
      <Text style={[styles.sectionTitle, { color: c.text }]}>Leaderboards</Text>
      
      <View style={styles.leaderboardContainer}>
        {gameScenarios.map((scenario) => (
          <View key={scenario.id} style={[styles.leaderboardCard, { backgroundColor: c.surface }]}>
            <Text style={[styles.leaderboardTitle, { color: c.text }]}>{scenario.name}</Text>
            
            <View style={styles.leaderboardList}>
              {scenario.leaderboard.map((entry) => (
                <View key={entry.rank} style={[
                  styles.leaderboardEntry,
                  entry.isCurrentUser && { backgroundColor: c.primary + '20' }
                ]}>
                  <View style={styles.rankContainer}>
                    <Text style={[styles.rank, { color: c.text }]}>#{entry.rank}</Text>
                    {entry.rank <= 3 && (
                      <Icon 
                        name={entry.rank === 1 ? 'trophy' : entry.rank === 2 ? 'medal' : 'award'} 
                        size={16} 
                        color={entry.rank === 1 ? '#FFD700' : entry.rank === 2 ? '#C0C0C0' : '#CD7F32'} 
                      />
                    )}
                  </View>
                  <Text style={[styles.playerName, { color: c.text }]}>{entry.playerName}</Text>
                  <View style={styles.scoreContainer}>
                    <Text style={[styles.score, { color: c.primary }]}>{entry.score}</Text>
                    <Text style={[styles.return, { color: c.success }]}>+{entry.return}%</Text>
                  </View>
                </View>
              ))}
            </View>
          </View>
        ))}
      </View>
    </View>
  );

  return (
    <View style={[styles.container, { backgroundColor: c.background }]}>
      {/* Header */}
      <View style={[styles.header, { borderBottomColor: c.border }]}>
        <TouchableOpacity onPress={onBack} style={styles.backButton}>
          <Icon name="arrow-left" size={24} color={c.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: c.text }]}>Game Mode</Text>
        <View style={styles.headerActions}>
          <TouchableOpacity style={styles.headerAction}>
            <Icon name="trophy" size={24} color={c.textMuted} />
          </TouchableOpacity>
        </View>
      </View>

      {/* Tabs */}
      <View style={[styles.tabsContainer, { backgroundColor: c.surface }]}>
        {[
          { id: 'scenarios', label: 'Scenarios', icon: 'view-grid' },
          { id: 'active', label: 'Active', icon: 'play' },
          { id: 'leaderboard', label: 'Leaderboard', icon: 'trophy' },
          { id: 'replay', label: 'Replay', icon: 'history' }
        ].map((tab) => (
          <TouchableOpacity
            key={tab.id}
            style={[
              styles.tab,
              activeTab === tab.id && { backgroundColor: c.primary }
            ]}
            onPress={() => setActiveTab(tab.id as any)}
          >
            <Icon 
              name={tab.icon} 
              size={20} 
              color={activeTab === tab.id ? c.background : c.textMuted} 
            />
            <Text style={[
              styles.tabText,
              { color: activeTab === tab.id ? c.background : c.textMuted }
            ]}>
              {tab.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Content */}
      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {activeTab === 'scenarios' && renderScenarios()}
        {activeTab === 'active' && renderActiveGame()}
        {activeTab === 'leaderboard' && renderLeaderboard()}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '700',
    flex: 1,
    textAlign: 'center',
  },
  headerActions: {
    flexDirection: 'row',
    gap: 8,
  },
  headerAction: {
    padding: 8,
  },
  tabsContainer: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    paddingVertical: 12,
    gap: 8,
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    gap: 8,
  },
  tabText: {
    fontSize: 14,
    fontWeight: '600',
  },
  content: {
    flex: 1,
    padding: 20,
  },
  tabContent: {
    flex: 1,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '700',
    marginBottom: 8,
  },
  sectionDescription: {
    fontSize: 16,
    marginBottom: 24,
    lineHeight: 24,
  },
  scenariosGrid: {
    gap: 16,
  },
  scenarioCard: {
    padding: 20,
    borderRadius: 12,
  },
  scenarioHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  scenarioName: {
    fontSize: 18,
    fontWeight: '700',
    flex: 1,
  },
  difficultyBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  difficultyText: {
    fontSize: 10,
    fontWeight: '700',
  },
  scenarioDescription: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 16,
  },
  scenarioInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  infoItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  infoText: {
    fontSize: 12,
  },
  scenarioChallenges: {
    marginBottom: 16,
  },
  challengesTitle: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 8,
  },
  challengeText: {
    fontSize: 12,
    marginBottom: 4,
  },
  scenarioActions: {
    flexDirection: 'row',
    gap: 12,
  },
  playButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: 8,
    gap: 8,
  },
  playButtonText: {
    fontSize: 14,
    fontWeight: '600',
  },
  previewButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: 8,
    borderWidth: 1,
    gap: 8,
  },
  previewButtonText: {
    fontSize: 14,
    fontWeight: '600',
  },
  gameContainer: {
    padding: 20,
    borderRadius: 12,
  },
  gameHeader: {
    marginBottom: 20,
  },
  gameTitle: {
    fontSize: 20,
    fontWeight: '700',
    marginBottom: 12,
  },
  gameProgress: {
    gap: 8,
  },
  progressText: {
    fontSize: 14,
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
  gameControls: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 20,
  },
  controlButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: 8,
    gap: 8,
  },
  controlText: {
    fontSize: 14,
    fontWeight: '600',
  },
  gameStats: {
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
    fontSize: 16,
    fontWeight: '700',
  },
  noActiveGame: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 60,
  },
  noGameText: {
    fontSize: 18,
    fontWeight: '600',
    marginTop: 16,
  },
  noGameSubtext: {
    fontSize: 14,
    marginTop: 8,
    textAlign: 'center',
  },
  leaderboardContainer: {
    gap: 16,
  },
  leaderboardCard: {
    padding: 16,
    borderRadius: 12,
  },
  leaderboardTitle: {
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 12,
  },
  leaderboardList: {
    gap: 8,
  },
  leaderboardEntry: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
  },
  rankContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    width: 60,
    gap: 4,
  },
  rank: {
    fontSize: 14,
    fontWeight: '700',
  },
  playerName: {
    flex: 1,
    fontSize: 14,
    fontWeight: '600',
  },
  scoreContainer: {
    alignItems: 'flex-end',
  },
  score: {
    fontSize: 14,
    fontWeight: '700',
  },
  return: {
    fontSize: 12,
    fontWeight: '600',
  },
});

export default GameModeScreen;
