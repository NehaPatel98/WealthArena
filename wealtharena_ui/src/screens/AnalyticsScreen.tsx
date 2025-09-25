import React, { useState } from 'react';
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

interface PerformanceMetric {
  name: string;
  value: number;
  change: number;
  changePercent: number;
  trend: 'up' | 'down' | 'neutral';
}

interface Report {
  id: string;
  name: string;
  type: 'performance' | 'risk' | 'attribution' | 'stress';
  date: string;
  status: 'ready' | 'generating' | 'error';
}

interface AnalyticsScreenProps {
  userProfile: {
    experience: 'beginner' | 'intermediate' | 'advanced';
  };
  onBack: () => void;
}

const AnalyticsScreen: React.FC<AnalyticsScreenProps> = ({ userProfile, onBack }) => {
  const [activeTab, setActiveTab] = useState<'performance' | 'risk' | 'attribution' | 'reports'>('performance');
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);
  
  const isDarkMode = true;
  const c = colors[isDarkMode ? 'dark' : 'light'];

  const performanceMetrics: PerformanceMetric[] = [
    {
      name: 'Total Return',
      value: 24.7,
      change: 2.3,
      changePercent: 10.3,
      trend: 'up'
    },
    {
      name: 'Annualized Return',
      value: 12.3,
      change: 1.1,
      changePercent: 9.8,
      trend: 'up'
    },
    {
      name: 'Sharpe Ratio',
      value: 1.65,
      change: 0.15,
      changePercent: 10.0,
      trend: 'up'
    },
    {
      name: 'Max Drawdown',
      value: -8.2,
      change: -0.5,
      changePercent: -6.5,
      trend: 'down'
    },
    {
      name: 'Volatility',
      value: 18.5,
      change: -1.2,
      changePercent: -6.1,
      trend: 'down'
    },
    {
      name: 'Beta',
      value: 0.95,
      change: 0.05,
      changePercent: 5.6,
      trend: 'up'
    }
  ];

  const riskMetrics = [
    { name: 'VaR (95%)', value: '-2.3%', color: c.danger },
    { name: 'CVaR (95%)', value: '-3.1%', color: c.danger },
    { name: 'Tracking Error', value: '4.2%', color: c.warning },
    { name: 'Information Ratio', value: '0.85', color: c.success },
    { name: 'Sortino Ratio', value: '1.42', color: c.success },
    { name: 'Calmar Ratio', value: '1.50', color: c.success }
  ];

  const attributionData = [
    { factor: 'Stock Selection', contribution: 8.2, color: c.primary },
    { factor: 'Asset Allocation', contribution: 5.1, color: c.success },
    { factor: 'Currency', contribution: 1.8, color: c.warning },
    { factor: 'Interest Rate', contribution: -0.5, color: c.danger },
    { factor: 'Other', contribution: 2.1, color: c.textMuted }
  ];

  const reports: Report[] = [
    {
      id: '1',
      name: 'Monthly Performance Report',
      type: 'performance',
      date: '2024-01-31',
      status: 'ready'
    },
    {
      id: '2',
      name: 'Risk Analysis Report',
      type: 'risk',
      date: '2024-01-31',
      status: 'ready'
    },
    {
      id: '3',
      name: 'Factor Attribution Report',
      type: 'attribution',
      date: '2024-01-31',
      status: 'generating'
    },
    {
      id: '4',
      name: 'Stress Test Results',
      type: 'stress',
      date: '2024-01-31',
      status: 'ready'
    }
  ];

  const generateReport = (reportType: string) => {
    Alert.alert(
      'Generate Report',
      `Generating ${reportType} report... This may take a few moments.`,
      [{ text: 'OK' }]
    );
  };

  const renderPerformance = () => (
    <View style={styles.tabContent}>
      <Text style={[styles.sectionTitle, { color: c.text }]}>Performance Analytics</Text>
      
      <View style={styles.metricsGrid}>
        {performanceMetrics.map((metric, index) => (
          <View key={index} style={[styles.metricCard, { backgroundColor: c.surface }]}>
            <Text style={[styles.metricName, { color: c.text }]}>{metric.name}</Text>
            <Text style={[styles.metricValue, { color: c.text }]}>
              {metric.value}{metric.name.includes('Ratio') ? '' : '%'}
            </Text>
            <View style={[
              styles.metricChange,
              { backgroundColor: metric.trend === 'up' ? c.success + '20' : 
                               metric.trend === 'down' ? c.danger + '20' : c.textMuted + '20' }
            ]}>
              <Icon 
                name={metric.trend === 'up' ? 'trending-up' : 
                      metric.trend === 'down' ? 'trending-down' : 'minus'} 
                size={12} 
                color={metric.trend === 'up' ? c.success : 
                       metric.trend === 'down' ? c.danger : c.textMuted} 
              />
              <Text style={[
                styles.changeText,
                { color: metric.trend === 'up' ? c.success : 
                         metric.trend === 'down' ? c.danger : c.textMuted }
              ]}>
                {metric.changePercent > 0 ? '+' : ''}{metric.changePercent.toFixed(1)}%
              </Text>
            </View>
          </View>
        ))}
      </View>
      
      <View style={[styles.chartContainer, { backgroundColor: c.surface }]}>
        <Text style={[styles.chartTitle, { color: c.text }]}>Performance Chart</Text>
        <View style={[styles.chartPlaceholder, { backgroundColor: c.background }]}>
          <Icon name="chart-line" size={48} color={c.textMuted} />
          <Text style={[styles.chartLabel, { color: c.textMuted }]}>Equity Curve</Text>
        </View>
      </View>
    </View>
  );

  const renderRisk = () => (
    <View style={styles.tabContent}>
      <Text style={[styles.sectionTitle, { color: c.text }]}>Risk Analytics</Text>
      
      <View style={styles.riskMetricsGrid}>
        {riskMetrics.map((metric, index) => (
          <View key={index} style={[styles.riskMetricCard, { backgroundColor: c.surface }]}>
            <Text style={[styles.riskMetricName, { color: c.text }]}>{metric.name}</Text>
            <Text style={[styles.riskMetricValue, { color: metric.color }]}>
              {metric.value}
            </Text>
          </View>
        ))}
      </View>
      
      <View style={[styles.riskChartContainer, { backgroundColor: c.surface }]}>
        <Text style={[styles.chartTitle, { color: c.text }]}>Risk-Return Scatter</Text>
        <View style={[styles.chartPlaceholder, { backgroundColor: c.background }]}>
          <Icon name="chart-scatter-plot" size={48} color={c.textMuted} />
          <Text style={[styles.chartLabel, { color: c.textMuted }]}>Risk vs Return</Text>
        </View>
      </View>
    </View>
  );

  const renderAttribution = () => (
    <View style={styles.tabContent}>
      <Text style={[styles.sectionTitle, { color: c.text }]}>Factor Attribution</Text>
      
      <View style={styles.attributionContainer}>
        {attributionData.map((item, index) => (
          <View key={index} style={styles.attributionItem}>
            <View style={styles.attributionInfo}>
              <View style={[styles.attributionDot, { backgroundColor: item.color }]} />
              <Text style={[styles.attributionFactor, { color: c.text }]}>{item.factor}</Text>
            </View>
            <Text style={[styles.attributionContribution, { color: c.text }]}>
              {item.contribution > 0 ? '+' : ''}{item.contribution}%
            </Text>
          </View>
        ))}
      </View>
      
      <View style={[styles.attributionChart, { backgroundColor: c.surface }]}>
        <Text style={[styles.chartTitle, { color: c.text }]}>Attribution Breakdown</Text>
        <View style={[styles.chartPlaceholder, { backgroundColor: c.background }]}>
          <Icon name="chart-pie" size={48} color={c.textMuted} />
          <Text style={[styles.chartLabel, { color: c.textMuted }]}>Factor Contributions</Text>
        </View>
      </View>
    </View>
  );

  const renderReports = () => (
    <View style={styles.tabContent}>
      <Text style={[styles.sectionTitle, { color: c.text }]}>Reports</Text>
      
      <View style={styles.reportsList}>
        {reports.map((report) => (
          <View key={report.id} style={[styles.reportCard, { backgroundColor: c.surface }]}>
            <View style={styles.reportHeader}>
              <Text style={[styles.reportName, { color: c.text }]}>{report.name}</Text>
              <View style={[
                styles.statusBadge,
                { backgroundColor: report.status === 'ready' ? c.success : 
                                 report.status === 'generating' ? c.warning : c.danger }
              ]}>
                <Text style={[styles.statusText, { color: c.background }]}>
                  {report.status.toUpperCase()}
                </Text>
              </View>
            </View>
            
            <Text style={[styles.reportDate, { color: c.textMuted }]}>
              Generated: {report.date}
            </Text>
            
            <View style={styles.reportActions}>
              {report.status === 'ready' ? (
                <>
                  <TouchableOpacity style={[styles.actionButton, { backgroundColor: c.primary }]}>
                    <Icon name="download" size={16} color={c.background} />
                    <Text style={[styles.actionText, { color: c.background }]}>Download</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={[styles.actionButton, { backgroundColor: c.background, borderColor: c.border }]}>
                    <Icon name="eye" size={16} color={c.text} />
                    <Text style={[styles.actionText, { color: c.text }]}>Preview</Text>
                  </TouchableOpacity>
                </>
              ) : (
                <TouchableOpacity 
                  style={[styles.actionButton, { backgroundColor: c.warning }]}
                  onPress={() => generateReport(report.name)}
                >
                  <Icon name="refresh" size={16} color={c.background} />
                  <Text style={[styles.actionText, { color: c.background }]}>Generate</Text>
                </TouchableOpacity>
              )}
            </View>
          </View>
        ))}
      </View>
      
      <TouchableOpacity 
        style={[styles.generateAllButton, { backgroundColor: c.primary }]}
        onPress={() => generateReport('All Reports')}
      >
        <Icon name="file-document-multiple" size={20} color={c.background} />
        <Text style={[styles.generateAllText, { color: c.background }]}>
          Generate All Reports
        </Text>
      </TouchableOpacity>
    </View>
  );

  return (
    <View style={[styles.container, { backgroundColor: c.background }]}>
      {/* Header */}
      <View style={[styles.header, { borderBottomColor: c.border }]}>
        <TouchableOpacity onPress={onBack} style={styles.backButton}>
          <Icon name="arrow-left" size={24} color={c.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: c.text }]}>Analytics & Reports</Text>
        <View style={styles.headerActions}>
          <TouchableOpacity style={styles.headerAction}>
            <Icon name="download" size={24} color={c.textMuted} />
          </TouchableOpacity>
        </View>
      </View>

      {/* Tabs */}
      <View style={[styles.tabsContainer, { backgroundColor: c.surface }]}>
        {[
          { id: 'performance', label: 'Performance', icon: 'chart-line' },
          { id: 'risk', label: 'Risk', icon: 'shield-alert' },
          { id: 'attribution', label: 'Attribution', icon: 'chart-pie' },
          { id: 'reports', label: 'Reports', icon: 'file-document' }
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
        {activeTab === 'performance' && renderPerformance()}
        {activeTab === 'risk' && renderRisk()}
        {activeTab === 'attribution' && renderAttribution()}
        {activeTab === 'reports' && renderReports()}
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
    marginBottom: 20,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginBottom: 20,
  },
  metricCard: {
    flex: 1,
    minWidth: '45%',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  metricName: {
    fontSize: 14,
    marginBottom: 8,
    textAlign: 'center',
  },
  metricValue: {
    fontSize: 20,
    fontWeight: '700',
    marginBottom: 8,
  },
  metricChange: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    gap: 4,
  },
  changeText: {
    fontSize: 12,
    fontWeight: '600',
  },
  chartContainer: {
    padding: 16,
    borderRadius: 12,
  },
  chartTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  chartPlaceholder: {
    height: 200,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 8,
  },
  chartLabel: {
    fontSize: 14,
    marginTop: 8,
  },
  riskMetricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginBottom: 20,
  },
  riskMetricCard: {
    flex: 1,
    minWidth: '30%',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  riskMetricName: {
    fontSize: 12,
    marginBottom: 8,
    textAlign: 'center',
  },
  riskMetricValue: {
    fontSize: 18,
    fontWeight: '700',
  },
  riskChartContainer: {
    padding: 16,
    borderRadius: 12,
  },
  attributionContainer: {
    gap: 12,
    marginBottom: 20,
  },
  attributionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
    paddingHorizontal: 16,
    backgroundColor: '#333',
    borderRadius: 8,
  },
  attributionInfo: {
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
    fontSize: 16,
    fontWeight: '600',
  },
  attributionContribution: {
    fontSize: 16,
    fontWeight: '700',
  },
  attributionChart: {
    padding: 16,
    borderRadius: 12,
  },
  reportsList: {
    gap: 16,
    marginBottom: 20,
  },
  reportCard: {
    padding: 16,
    borderRadius: 12,
  },
  reportHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  reportName: {
    fontSize: 16,
    fontWeight: '700',
    flex: 1,
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  statusText: {
    fontSize: 10,
    fontWeight: '700',
  },
  reportDate: {
    fontSize: 14,
    marginBottom: 12,
  },
  reportActions: {
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
  generateAllButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    borderRadius: 12,
    gap: 12,
  },
  generateAllText: {
    fontSize: 16,
    fontWeight: '700',
  },
});

export default AnalyticsScreen;
