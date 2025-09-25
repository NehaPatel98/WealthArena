import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Modal,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { colors } from '../theme/colors';

interface StrategyTemplate {
  id: string;
  name: string;
  description: string;
  category: 'momentum' | 'mean-reversion' | 'arbitrage' | 'ml' | 'custom';
  complexity: 'beginner' | 'intermediate' | 'advanced';
  expectedReturn: number;
  maxDrawdown: number;
  sharpeRatio: number;
}

interface StrategyLabScreenProps {
  userProfile: {
    experience: 'beginner' | 'intermediate' | 'advanced';
  };
  onBack: () => void;
}

const StrategyLabScreen: React.FC<StrategyLabScreenProps> = ({ userProfile, onBack }) => {
  const [activeTab, setActiveTab] = useState<'templates' | 'builder' | 'backtest' | 'comparison'>('templates');
  const [selectedTemplate, setSelectedTemplate] = useState<StrategyTemplate | null>(null);
  const [showParameterModal, setShowParameterModal] = useState(false);
  
  const isDarkMode = true;
  const c = colors[isDarkMode ? 'dark' : 'light'];

  const strategyTemplates: StrategyTemplate[] = [
    {
      id: 'momentum-1',
      name: 'Simple Momentum',
      description: 'Buy when price breaks above 20-day high, sell when below 20-day low',
      category: 'momentum',
      complexity: 'beginner',
      expectedReturn: 12.5,
      maxDrawdown: 15.2,
      sharpeRatio: 1.8,
    },
    {
      id: 'mean-reversion-1',
      name: 'Bollinger Band Mean Reversion',
      description: 'Buy when price touches lower band, sell when touches upper band',
      category: 'mean-reversion',
      complexity: 'intermediate',
      expectedReturn: 8.7,
      maxDrawdown: 12.1,
      sharpeRatio: 1.4,
    },
    {
      id: 'ml-1',
      name: 'Machine Learning Predictor',
      description: 'Uses LSTM neural network to predict price movements',
      category: 'ml',
      complexity: 'advanced',
      expectedReturn: 15.3,
      maxDrawdown: 18.7,
      sharpeRatio: 2.1,
    }
  ];

  const handleTemplateSelect = (template: StrategyTemplate) => {
    setSelectedTemplate(template);
    setShowParameterModal(true);
  };

  const renderTemplates = () => (
    <View style={styles.tabContent}>
      <Text style={[styles.sectionTitle, { color: c.text }]}>Strategy Templates</Text>
      <Text style={[styles.sectionDescription, { color: c.textMuted }]}>
        Choose from pre-built strategies or create your own
      </Text>
      
      <View style={styles.templatesGrid}>
        {strategyTemplates.map((template) => (
          <TouchableOpacity
            key={template.id}
            style={[styles.templateCard, { backgroundColor: c.surface }]}
            onPress={() => handleTemplateSelect(template)}
          >
            <View style={styles.templateHeader}>
              <Text style={[styles.templateName, { color: c.text }]}>{template.name}</Text>
              <View style={[styles.complexityBadge, { 
                backgroundColor: template.complexity === 'beginner' ? c.success : 
                               template.complexity === 'intermediate' ? c.warning : c.danger 
              }]}>
                <Text style={[styles.complexityText, { color: c.background }]}>
                  {template.complexity.toUpperCase()}
                </Text>
              </View>
            </View>
            
            <Text style={[styles.templateDescription, { color: c.textMuted }]}>
              {template.description}
            </Text>
            
            <View style={styles.templateMetrics}>
              <View style={styles.metricItem}>
                <Text style={[styles.metricLabel, { color: c.textMuted }]}>Expected Return</Text>
                <Text style={[styles.metricValue, { color: c.success }]}>{template.expectedReturn}%</Text>
              </View>
              <View style={styles.metricItem}>
                <Text style={[styles.metricLabel, { color: c.textMuted }]}>Max Drawdown</Text>
                <Text style={[styles.metricValue, { color: c.danger }]}>{template.maxDrawdown}%</Text>
              </View>
              <View style={styles.metricItem}>
                <Text style={[styles.metricLabel, { color: c.textMuted }]}>Sharpe Ratio</Text>
                <Text style={[styles.metricValue, { color: c.primary }]}>{template.sharpeRatio}</Text>
              </View>
            </View>
            
            <View style={styles.templateActions}>
              <TouchableOpacity style={[styles.actionButton, { backgroundColor: c.primary }]}>
                <Icon name="play" size={16} color={c.background} />
                <Text style={[styles.actionText, { color: c.background }]}>Backtest</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.actionButton, { backgroundColor: c.background, borderColor: c.border }]}>
                <Icon name="eye" size={16} color={c.text} />
                <Text style={[styles.actionText, { color: c.text }]}>Preview</Text>
              </TouchableOpacity>
            </View>
          </TouchableOpacity>
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
        <Text style={[styles.headerTitle, { color: c.text }]}>Strategy Lab</Text>
        <View style={styles.headerActions}>
          <TouchableOpacity style={styles.headerAction}>
            <Icon name="help-circle" size={24} color={c.textMuted} />
          </TouchableOpacity>
        </View>
      </View>

      {/* Tabs */}
      <View style={[styles.tabsContainer, { backgroundColor: c.surface }]}>
        {[
          { id: 'templates', label: 'Templates', icon: 'view-grid' },
          { id: 'builder', label: 'Builder', icon: 'flask' },
          { id: 'backtest', label: 'Backtest', icon: 'chart-line' },
          { id: 'comparison', label: 'Compare', icon: 'scale-balance' }
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
        {activeTab === 'templates' && renderTemplates()}
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
  templatesGrid: {
    gap: 16,
  },
  templateCard: {
    padding: 20,
    borderRadius: 12,
  },
  templateHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  templateName: {
    fontSize: 18,
    fontWeight: '700',
    flex: 1,
  },
  complexityBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  complexityText: {
    fontSize: 10,
    fontWeight: '700',
  },
  templateDescription: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 16,
  },
  templateMetrics: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  metricItem: {
    alignItems: 'center',
  },
  metricLabel: {
    fontSize: 12,
    marginBottom: 4,
  },
  metricValue: {
    fontSize: 16,
    fontWeight: '700',
  },
  templateActions: {
    flexDirection: 'row',
    gap: 12,
  },
  actionButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: 8,
    gap: 8,
    borderWidth: 1,
  },
  actionText: {
    fontSize: 14,
    fontWeight: '600',
  },
});

export default StrategyLabScreen;