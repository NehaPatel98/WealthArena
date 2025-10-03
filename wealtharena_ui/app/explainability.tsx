import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { Stack } from 'expo-router';
import {
  FileText,
  TrendingUp,
  AlertCircle,
  Download,
  Eye,
  Brain,
  Activity,
  ChevronDown,
  ChevronUp,
} from 'lucide-react-native';
import Colors from '@/constants/colors';

interface TradeRationale {
  id: string;
  symbol: string;
  action: 'buy' | 'sell';
  timestamp: Date;
  reasoning: string;
  confidence: number;
  signals: string[];
  news: string[];
}

const TRADE_RATIONALES: TradeRationale[] = [
  {
    id: '1',
    symbol: 'AAPL',
    action: 'buy',
    timestamp: new Date('2025-01-15T10:30:00'),
    reasoning: 'Strong momentum indicators combined with positive earnings outlook. RSI shows oversold conditions with bullish divergence.',
    confidence: 87,
    signals: ['RSI Oversold', 'MACD Bullish Cross', 'Volume Surge'],
    news: ['Q4 Earnings Beat Expectations', 'New Product Launch Announced'],
  },
  {
    id: '2',
    symbol: 'TSLA',
    action: 'sell',
    timestamp: new Date('2025-01-14T14:20:00'),
    reasoning: 'Overbought conditions detected. Price reached resistance level with declining volume, suggesting potential reversal.',
    confidence: 72,
    signals: ['RSI Overbought', 'Resistance Level', 'Declining Volume'],
    news: ['Production Concerns Raised', 'Analyst Downgrade'],
  },
];

const FEATURE_ATTRIBUTION = [
  { feature: 'Price Momentum', importance: 0.28, value: '+2.3σ' },
  { feature: 'Volume Profile', importance: 0.22, value: 'High' },
  { feature: 'Market Sentiment', importance: 0.18, value: 'Bullish' },
  { feature: 'Technical Indicators', importance: 0.15, value: 'Strong' },
  { feature: 'News Sentiment', importance: 0.12, value: 'Positive' },
  { feature: 'Sector Performance', importance: 0.05, value: 'Neutral' },
];

export default function ExplainabilityScreen() {
  const [expandedTrade, setExpandedTrade] = useState<string | null>(null);

  const toggleTrade = (id: string) => {
    setExpandedTrade(expandedTrade === id ? null : id);
  };

  return (
    <View style={styles.container}>
      <Stack.Screen
        options={{
          title: 'Explainability & Audit',
          headerStyle: { backgroundColor: Colors.background },
          headerTintColor: Colors.text,
        }}
      />
      <ScrollView showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <View style={styles.headerIcon}>
            <Brain size={32} color={Colors.accent} />
          </View>
          <Text style={styles.headerTitle}>AI Decision Transparency</Text>
          <Text style={styles.headerSubtitle}>
            Understand every trade decision with detailed explanations
          </Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Trade Rationales</Text>
          {TRADE_RATIONALES.map((trade) => {
            const isExpanded = expandedTrade === trade.id;
            return (
              <View key={trade.id} style={styles.rationaleCard}>
                <TouchableOpacity
                  style={styles.rationaleHeader}
                  onPress={() => toggleTrade(trade.id)}
                >
                  <View style={styles.rationaleLeft}>
                    <View
                      style={[
                        styles.actionBadge,
                        {
                          backgroundColor:
                            trade.action === 'buy'
                              ? Colors.chartGreen + '20'
                              : Colors.chartRed + '20',
                        },
                      ]}
                    >
                      <TrendingUp
                        size={16}
                        color={trade.action === 'buy' ? Colors.chartGreen : Colors.chartRed}
                      />
                    </View>
                    <View>
                      <Text style={styles.rationaleSymbol}>{trade.symbol}</Text>
                      <Text style={styles.rationaleTime}>
                        {trade.timestamp.toLocaleDateString()} {trade.timestamp.toLocaleTimeString()}
                      </Text>
                    </View>
                  </View>
                  <View style={styles.rationaleRight}>
                    <View style={styles.confidenceBadge}>
                      <Text style={styles.confidenceText}>{trade.confidence}%</Text>
                    </View>
                    {isExpanded ? (
                      <ChevronUp size={20} color={Colors.textMuted} />
                    ) : (
                      <ChevronDown size={20} color={Colors.textMuted} />
                    )}
                  </View>
                </TouchableOpacity>

                {isExpanded && (
                  <View style={styles.rationaleContent}>
                    <View style={styles.reasoningSection}>
                      <Text style={styles.reasoningLabel}>Reasoning</Text>
                      <Text style={styles.reasoningText}>{trade.reasoning}</Text>
                    </View>

                    <View style={styles.signalsSection}>
                      <Text style={styles.signalsLabel}>Key Signals</Text>
                      <View style={styles.signalsGrid}>
                        {trade.signals.map((signal, index) => (
                          <View key={index} style={styles.signalChip}>
                            <Activity size={12} color={Colors.secondary} />
                            <Text style={styles.signalText}>{signal}</Text>
                          </View>
                        ))}
                      </View>
                    </View>

                    <View style={styles.newsSection}>
                      <Text style={styles.newsLabel}>Supporting News</Text>
                      {trade.news.map((item, index) => (
                        <View key={index} style={styles.newsItem}>
                          <FileText size={14} color={Colors.textMuted} />
                          <Text style={styles.newsText}>{item}</Text>
                        </View>
                      ))}
                    </View>
                  </View>
                )}
              </View>
            );
          })}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Feature Attribution</Text>
          <View style={styles.attributionCard}>
            <Text style={styles.attributionSubtitle}>
              Model decision factors ranked by importance
            </Text>
            {FEATURE_ATTRIBUTION.map((item, index) => (
              <View key={index} style={styles.attributionRow}>
                <View style={styles.attributionLeft}>
                  <Text style={styles.attributionFeature}>{item.feature}</Text>
                  <View style={styles.attributionBar}>
                    <View
                      style={[
                        styles.attributionFill,
                        { width: `${item.importance * 100}%` },
                      ]}
                    />
                  </View>
                </View>
                <View style={styles.attributionRight}>
                  <Text style={styles.attributionValue}>{item.value}</Text>
                  <Text style={styles.attributionImportance}>
                    {(item.importance * 100).toFixed(0)}%
                  </Text>
                </View>
              </View>
            ))}
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Decision Trace</Text>
          <View style={styles.traceCard}>
            <View style={styles.traceRow}>
              <Eye size={20} color={Colors.secondary} />
              <View style={styles.traceContent}>
                <Text style={styles.traceLabel}>Model Version</Text>
                <Text style={styles.traceValue}>v2.4.1 (Production)</Text>
              </View>
            </View>
            <View style={styles.traceDivider} />
            <View style={styles.traceRow}>
              <Brain size={20} color={Colors.accent} />
              <View style={styles.traceContent}>
                <Text style={styles.traceLabel}>Training Dataset</Text>
                <Text style={styles.traceValue}>Market Data 2020-2024</Text>
              </View>
            </View>
            <View style={styles.traceDivider} />
            <View style={styles.traceRow}>
              <Activity size={20} color={Colors.gold} />
              <View style={styles.traceContent}>
                <Text style={styles.traceLabel}>Last Updated</Text>
                <Text style={styles.traceValue}>2025-01-10 08:00 UTC</Text>
              </View>
            </View>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Export Audit Bundle</Text>
          <View style={styles.exportCard}>
            <AlertCircle size={24} color={Colors.warning} />
            <View style={styles.exportContent}>
              <Text style={styles.exportTitle}>Compliance Ready</Text>
              <Text style={styles.exportSubtitle}>
                Export complete audit trail with all decision rationales
              </Text>
            </View>
          </View>
          <View style={styles.exportButtons}>
            <TouchableOpacity style={styles.exportButton}>
              <Download size={20} color={Colors.text} />
              <Text style={styles.exportButtonText}>Export CSV</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.exportButton}>
              <Download size={20} color={Colors.text} />
              <Text style={styles.exportButtonText}>Export PDF</Text>
            </TouchableOpacity>
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
  header: {
    padding: 24,
    alignItems: 'center',
  },
  headerIcon: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: Colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: '700' as const,
    color: Colors.text,
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 14,
    color: Colors.textSecondary,
    textAlign: 'center',
  },
  section: {
    padding: 24,
    paddingTop: 0,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '700' as const,
    color: Colors.text,
    marginBottom: 16,
  },
  rationaleCard: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    marginBottom: 12,
    overflow: 'hidden',
  },
  rationaleHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
  },
  rationaleLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    flex: 1,
  },
  actionBadge: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  rationaleSymbol: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  rationaleTime: {
    fontSize: 12,
    color: Colors.textMuted,
    marginTop: 2,
  },
  rationaleRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  confidenceBadge: {
    backgroundColor: Colors.accent + '20',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  confidenceText: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.accent,
  },
  rationaleContent: {
    padding: 16,
    paddingTop: 0,
    gap: 16,
  },
  reasoningSection: {
    gap: 8,
  },
  reasoningLabel: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.textSecondary,
  },
  reasoningText: {
    fontSize: 14,
    color: Colors.text,
    lineHeight: 20,
  },
  signalsSection: {
    gap: 8,
  },
  signalsLabel: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.textSecondary,
  },
  signalsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  signalChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: Colors.surfaceLight,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  signalText: {
    fontSize: 12,
    color: Colors.text,
  },
  newsSection: {
    gap: 8,
  },
  newsLabel: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.textSecondary,
  },
  newsItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
  },
  newsText: {
    flex: 1,
    fontSize: 13,
    color: Colors.text,
    lineHeight: 18,
  },
  attributionCard: {
    backgroundColor: Colors.surface,
    padding: 20,
    borderRadius: 16,
    gap: 16,
  },
  attributionSubtitle: {
    fontSize: 13,
    color: Colors.textSecondary,
    marginBottom: 8,
  },
  attributionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 16,
  },
  attributionLeft: {
    flex: 1,
    gap: 8,
  },
  attributionFeature: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  attributionBar: {
    height: 6,
    backgroundColor: Colors.surfaceLight,
    borderRadius: 3,
    overflow: 'hidden',
  },
  attributionFill: {
    height: '100%',
    backgroundColor: Colors.secondary,
    borderRadius: 3,
  },
  attributionRight: {
    alignItems: 'flex-end',
    gap: 4,
  },
  attributionValue: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  attributionImportance: {
    fontSize: 12,
    color: Colors.textMuted,
  },
  traceCard: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    overflow: 'hidden',
  },
  traceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    padding: 16,
  },
  traceContent: {
    flex: 1,
    gap: 4,
  },
  traceLabel: {
    fontSize: 13,
    color: Colors.textSecondary,
  },
  traceValue: {
    fontSize: 15,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  traceDivider: {
    height: 1,
    backgroundColor: Colors.border,
    marginHorizontal: 16,
  },
  exportCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    backgroundColor: Colors.surface,
    padding: 20,
    borderRadius: 16,
    marginBottom: 16,
  },
  exportContent: {
    flex: 1,
    gap: 4,
  },
  exportTitle: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  exportSubtitle: {
    fontSize: 13,
    color: Colors.textSecondary,
    lineHeight: 18,
  },
  exportButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  exportButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: Colors.secondary,
    paddingVertical: 14,
    borderRadius: 12,
  },
  exportButtonText: {
    fontSize: 15,
    fontWeight: '600' as const,
    color: Colors.text,
  },
});
