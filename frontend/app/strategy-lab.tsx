import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { apiClient, TradingSignal } from '../services/apiService';
import FloatingChatbot from '../components/FloatingChatbot';

interface Strategy {
  id: string;
  name: string;
  description: string;
  risk_level: 'low' | 'medium' | 'high';
  expected_return: number;
  max_drawdown: number;
  sharpe_ratio: number;
  is_active: boolean;
}

const StrategyLab = () => {
  const navigation = useNavigation();
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [signals, setSignals] = useState<TradingSignal[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newStrategy, setNewStrategy] = useState({
    name: '',
    description: '',
    risk_level: 'medium' as 'low' | 'medium' | 'high',
  });

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      
      // Load trading signals
      const signalsResponse = await apiClient.getTopSignals();
      if (signalsResponse.success) {
        setSignals(signalsResponse.data?.signals || []);
      }

      // Load user strategies (mock data for now)
      const mockStrategies: Strategy[] = [
        {
          id: '1',
          name: 'Conservative Growth',
          description: 'Low-risk strategy focusing on blue-chip stocks with steady dividends',
          risk_level: 'low',
          expected_return: 0.08,
          max_drawdown: 0.05,
          sharpe_ratio: 1.2,
          is_active: true,
        },
        {
          id: '2',
          name: 'Aggressive Growth',
          description: 'High-risk strategy targeting high-growth stocks and emerging markets',
          risk_level: 'high',
          expected_return: 0.15,
          max_drawdown: 0.20,
          sharpe_ratio: 0.8,
          is_active: false,
        },
        {
          id: '3',
          name: 'Balanced Portfolio',
          description: 'Moderate risk strategy with diversified asset allocation',
          risk_level: 'medium',
          expected_return: 0.12,
          max_drawdown: 0.10,
          sharpe_ratio: 1.0,
          is_active: true,
        },
      ];
      
      setStrategies(mockStrategies);
    } catch (error) {
      console.error('Error loading data:', error);
      Alert.alert('Error', 'Failed to load strategy data');
    } finally {
      setLoading(false);
    }
  };

  const createStrategy = () => {
    if (!newStrategy.name.trim()) {
      Alert.alert('Error', 'Please enter a strategy name');
      return;
    }

    const strategy: Strategy = {
      id: Date.now().toString(),
      name: newStrategy.name,
      description: newStrategy.description,
      risk_level: newStrategy.risk_level,
      expected_return: 0.10,
      max_drawdown: 0.08,
      sharpe_ratio: 1.0,
      is_active: false,
    };

    setStrategies([...strategies, strategy]);
    setNewStrategy({ name: '', description: '', risk_level: 'medium' });
    setShowCreateForm(false);
    Alert.alert('Success', 'Strategy created successfully');
  };

  const toggleStrategy = (strategyId: string) => {
    setStrategies(strategies.map(strategy => 
      strategy.id === strategyId 
        ? { ...strategy, is_active: !strategy.is_active }
        : strategy
    ));
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low': return '#4CAF50';
      case 'medium': return '#FF9800';
      case 'high': return '#F44336';
      default: return '#757575';
    }
  };

  const getRiskLabel = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low': return 'Low Risk';
      case 'medium': return 'Medium Risk';
      case 'high': return 'High Risk';
      default: return 'Unknown';
    }
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#6366F1" />
        <Text style={styles.loadingText}>Loading Strategy Lab...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#1E1B4B', '#312E81']}
        style={styles.header}
      >
        <View style={styles.headerContent}>
          <TouchableOpacity
            style={styles.backButton}
            onPress={() => navigation.goBack()}
          >
            <Ionicons name="arrow-back" size={24} color="white" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Strategy Lab</Text>
          <TouchableOpacity
            style={styles.createButton}
            onPress={() => setShowCreateForm(true)}
          >
            <Ionicons name="add" size={24} color="white" />
          </TouchableOpacity>
        </View>
      </LinearGradient>

      <ScrollView style={styles.content}>
        {/* AI Signals Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>AI Trading Signals</Text>
          <Text style={styles.sectionSubtitle}>Latest signals from our RL models</Text>
          
          {signals.length > 0 ? (
            <View style={styles.signalsContainer}>
              {signals.slice(0, 3).map((signal) => (
                <View key={signal.id} style={styles.signalCard}>
                  <View style={styles.signalHeader}>
                    <Text style={styles.signalSymbol}>{signal.symbol}</Text>
                    <View style={[
                      styles.signalType,
                      { backgroundColor: signal.signal_type === 'BUY' ? '#4CAF50' : '#F44336' }
                    ]}>
                      <Text style={styles.signalTypeText}>{signal.signal_type}</Text>
                    </View>
                  </View>
                  <Text style={styles.signalConfidence}>
                    Confidence: {(signal.confidence_score * 100).toFixed(1)}%
                  </Text>
                  <Text style={styles.signalPrice}>
                    Entry: ${signal.entry_price.toFixed(2)}
                  </Text>
                </View>
              ))}
            </View>
          ) : (
            <Text style={styles.noDataText}>No signals available</Text>
          )}
        </View>

        {/* User Strategies Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Your Strategies</Text>
          <Text style={styles.sectionSubtitle}>Create and manage your trading strategies</Text>
          
          {strategies.map((strategy) => (
            <View key={strategy.id} style={styles.strategyCard}>
              <View style={styles.strategyHeader}>
                <Text style={styles.strategyName}>{strategy.name}</Text>
                <TouchableOpacity
                  style={[
                    styles.toggleButton,
                    { backgroundColor: strategy.is_active ? '#4CAF50' : '#757575' }
                  ]}
                  onPress={() => toggleStrategy(strategy.id)}
                >
                  <Text style={styles.toggleButtonText}>
                    {strategy.is_active ? 'Active' : 'Inactive'}
                  </Text>
                </TouchableOpacity>
              </View>
              
              <Text style={styles.strategyDescription}>{strategy.description}</Text>
              
              <View style={styles.strategyMetrics}>
                <View style={styles.metric}>
                  <Text style={styles.metricLabel}>Risk Level</Text>
                  <Text style={[styles.metricValue, { color: getRiskColor(strategy.risk_level) }]}>
                    {getRiskLabel(strategy.risk_level)}
                  </Text>
                </View>
                <View style={styles.metric}>
                  <Text style={styles.metricLabel}>Expected Return</Text>
                  <Text style={styles.metricValue}>
                    {(strategy.expected_return * 100).toFixed(1)}%
                  </Text>
                </View>
                <View style={styles.metric}>
                  <Text style={styles.metricLabel}>Sharpe Ratio</Text>
                  <Text style={styles.metricValue}>
                    {strategy.sharpe_ratio.toFixed(2)}
                  </Text>
                </View>
              </View>
            </View>
          ))}
        </View>
      </ScrollView>

      {/* Create Strategy Modal */}
      {showCreateForm && (
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Create New Strategy</Text>
            
            <TextInput
              style={styles.input}
              placeholder="Strategy Name"
              value={newStrategy.name}
              onChangeText={(text) => setNewStrategy({ ...newStrategy, name: text })}
            />
            
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder="Description"
              value={newStrategy.description}
              onChangeText={(text) => setNewStrategy({ ...newStrategy, description: text })}
              multiline
              numberOfLines={3}
            />
            
            <View style={styles.riskSelector}>
              <Text style={styles.riskLabel}>Risk Level:</Text>
              <View style={styles.riskOptions}>
                {['low', 'medium', 'high'].map((level) => (
                  <TouchableOpacity
                    key={level}
                    style={[
                      styles.riskOption,
                      newStrategy.risk_level === level && styles.riskOptionSelected
                    ]}
                    onPress={() => setNewStrategy({ ...newStrategy, risk_level: level as any })}
                  >
                    <Text style={[
                      styles.riskOptionText,
                      newStrategy.risk_level === level && styles.riskOptionTextSelected
                    ]}>
                      {level.charAt(0).toUpperCase() + level.slice(1)}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
            
            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={styles.cancelButton}
                onPress={() => setShowCreateForm(false)}
              >
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.createButton}
                onPress={createStrategy}
              >
                <Text style={styles.createButtonText}>Create</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      )}

      <FloatingChatbot
        context={{
          current_page: 'strategy_lab',
          available_signals: signals.length,
          user_strategies: strategies.length,
        }}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F0F23',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0F0F23',
  },
  loadingText: {
    color: '#A1A1AA',
    marginTop: 16,
    fontSize: 16,
  },
  header: {
    paddingTop: 50,
    paddingBottom: 20,
    paddingHorizontal: 20,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
  },
  createButton: {
    padding: 8,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  section: {
    marginBottom: 30,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 8,
  },
  sectionSubtitle: {
    fontSize: 14,
    color: '#A1A1AA',
    marginBottom: 16,
  },
  signalsContainer: {
    gap: 12,
  },
  signalCard: {
    backgroundColor: '#1A1A2E',
    borderRadius: 12,
    padding: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#6366F1',
  },
  signalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  signalSymbol: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
  },
  signalType: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 16,
  },
  signalTypeText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  signalConfidence: {
    fontSize: 14,
    color: '#A1A1AA',
    marginBottom: 4,
  },
  signalPrice: {
    fontSize: 14,
    color: '#4CAF50',
    fontWeight: '600',
  },
  strategyCard: {
    backgroundColor: '#1A1A2E',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  strategyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  strategyName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
  },
  toggleButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  toggleButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  strategyDescription: {
    fontSize: 14,
    color: '#A1A1AA',
    marginBottom: 12,
  },
  strategyMetrics: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  metric: {
    alignItems: 'center',
  },
  metricLabel: {
    fontSize: 12,
    color: '#A1A1AA',
    marginBottom: 4,
  },
  metricValue: {
    fontSize: 14,
    fontWeight: 'bold',
    color: 'white',
  },
  noDataText: {
    color: '#A1A1AA',
    textAlign: 'center',
    fontSize: 16,
    marginTop: 20,
  },
  modalOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalContent: {
    backgroundColor: '#1A1A2E',
    borderRadius: 16,
    padding: 24,
    width: '100%',
    maxWidth: 400,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 20,
    textAlign: 'center',
  },
  input: {
    backgroundColor: '#2A2A3E',
    borderRadius: 8,
    padding: 12,
    color: 'white',
    fontSize: 16,
    marginBottom: 16,
  },
  textArea: {
    height: 80,
    textAlignVertical: 'top',
  },
  riskSelector: {
    marginBottom: 20,
  },
  riskLabel: {
    fontSize: 16,
    color: 'white',
    marginBottom: 12,
  },
  riskOptions: {
    flexDirection: 'row',
    gap: 8,
  },
  riskOption: {
    flex: 1,
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#2A2A3E',
    alignItems: 'center',
  },
  riskOptionSelected: {
    backgroundColor: '#6366F1',
  },
  riskOptionText: {
    color: '#A1A1AA',
    fontSize: 14,
    fontWeight: 'bold',
  },
  riskOptionTextSelected: {
    color: 'white',
  },
  modalButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  cancelButton: {
    flex: 1,
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#2A2A3E',
    alignItems: 'center',
  },
  cancelButtonText: {
    color: '#A1A1AA',
    fontSize: 16,
    fontWeight: 'bold',
  },
  createButton: {
    flex: 1,
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#6366F1',
    alignItems: 'center',
  },
  createButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default StrategyLab;