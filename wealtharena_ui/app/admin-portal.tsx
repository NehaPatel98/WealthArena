import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
} from 'react-native';
import { Stack } from 'expo-router';
import {
  Database,
  Key,
  Shield,
  BarChart3,
  Activity,
  Users,
  Settings,
  CheckCircle,
  AlertCircle,
  TrendingUp,
} from 'lucide-react-native';
import Colors from '@/constants/colors';

interface DataFeed {
  id: string;
  name: string;
  provider: string;
  status: 'active' | 'inactive' | 'error';
  lastSync: Date;
  recordsProcessed: number;
}

interface APIKey {
  id: string;
  name: string;
  key: string;
  created: Date;
  lastUsed: Date;
  requests: number;
}

const DATA_FEEDS: DataFeed[] = [
  {
    id: '1',
    name: 'Market Data Feed',
    provider: 'NYSE',
    status: 'active',
    lastSync: new Date('2025-01-15T10:30:00'),
    recordsProcessed: 125430,
  },
  {
    id: '2',
    name: 'News Sentiment',
    provider: 'Reuters',
    status: 'active',
    lastSync: new Date('2025-01-15T10:25:00'),
    recordsProcessed: 8920,
  },
  {
    id: '3',
    name: 'Options Data',
    provider: 'CBOE',
    status: 'error',
    lastSync: new Date('2025-01-15T09:15:00'),
    recordsProcessed: 0,
  },
  {
    id: '4',
    name: 'Crypto Prices',
    provider: 'Binance',
    status: 'inactive',
    lastSync: new Date('2025-01-14T18:00:00'),
    recordsProcessed: 45230,
  },
];

const API_KEYS: APIKey[] = [
  {
    id: '1',
    name: 'Production API',
    key: 'pk_live_••••••••••••1234',
    created: new Date('2024-12-01'),
    lastUsed: new Date('2025-01-15T10:30:00'),
    requests: 1245678,
  },
  {
    id: '2',
    name: 'Development API',
    key: 'pk_test_••••••••••••5678',
    created: new Date('2024-12-15'),
    lastUsed: new Date('2025-01-15T09:15:00'),
    requests: 45230,
  },
];

export default function AdminPortalScreen() {
  const [selectedTab, setSelectedTab] = useState<'feeds' | 'api' | 'compliance' | 'analytics'>('feeds');

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return Colors.accent;
      case 'inactive':
        return Colors.textMuted;
      case 'error':
        return Colors.danger;
      default:
        return Colors.textMuted;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <CheckCircle size={16} color={Colors.accent} />;
      case 'error':
        return <AlertCircle size={16} color={Colors.danger} />;
      default:
        return <Activity size={16} color={Colors.textMuted} />;
    }
  };

  return (
    <View style={styles.container}>
      <Stack.Screen
        options={{
          title: 'Admin Portal',
          headerStyle: { backgroundColor: Colors.background },
          headerTintColor: Colors.text,
        }}
      />
      <ScrollView showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <View style={styles.headerIcon}>
            <Shield size={32} color={Colors.gold} />
          </View>
          <Text style={styles.headerTitle}>Partner & Admin Portal</Text>
          <Text style={styles.headerSubtitle}>
            Manage data feeds, integrations, and compliance
          </Text>
        </View>

        <View style={styles.statsGrid}>
          <View style={styles.statCard}>
            <Database size={24} color={Colors.secondary} />
            <Text style={styles.statValue}>4</Text>
            <Text style={styles.statLabel}>Data Feeds</Text>
          </View>
          <View style={styles.statCard}>
            <Key size={24} color={Colors.accent} />
            <Text style={styles.statValue}>2</Text>
            <Text style={styles.statLabel}>API Keys</Text>
          </View>
          <View style={styles.statCard}>
            <Users size={24} color={Colors.gold} />
            <Text style={styles.statValue}>1.2M</Text>
            <Text style={styles.statLabel}>API Calls</Text>
          </View>
        </View>

        <View style={styles.tabBar}>
          <TouchableOpacity
            style={[styles.tab, selectedTab === 'feeds' && styles.tabActive]}
            onPress={() => setSelectedTab('feeds')}
          >
            <Database size={20} color={selectedTab === 'feeds' ? Colors.secondary : Colors.textMuted} />
            <Text style={[styles.tabText, selectedTab === 'feeds' && styles.tabTextActive]}>
              Data Feeds
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.tab, selectedTab === 'api' && styles.tabActive]}
            onPress={() => setSelectedTab('api')}
          >
            <Key size={20} color={selectedTab === 'api' ? Colors.secondary : Colors.textMuted} />
            <Text style={[styles.tabText, selectedTab === 'api' && styles.tabTextActive]}>
              API Keys
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.tab, selectedTab === 'compliance' && styles.tabActive]}
            onPress={() => setSelectedTab('compliance')}
          >
            <Shield size={20} color={selectedTab === 'compliance' ? Colors.secondary : Colors.textMuted} />
            <Text style={[styles.tabText, selectedTab === 'compliance' && styles.tabTextActive]}>
              Compliance
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.tab, selectedTab === 'analytics' && styles.tabActive]}
            onPress={() => setSelectedTab('analytics')}
          >
            <BarChart3 size={20} color={selectedTab === 'analytics' ? Colors.secondary : Colors.textMuted} />
            <Text style={[styles.tabText, selectedTab === 'analytics' && styles.tabTextActive]}>
              Analytics
            </Text>
          </TouchableOpacity>
        </View>

        {selectedTab === 'feeds' && (
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Data Feed Management</Text>
              <TouchableOpacity style={styles.addButton}>
                <Text style={styles.addButtonText}>+ Add Feed</Text>
              </TouchableOpacity>
            </View>
            {DATA_FEEDS.map((feed) => (
              <View key={feed.id} style={styles.feedCard}>
                <View style={styles.feedHeader}>
                  <View style={styles.feedLeft}>
                    <Text style={styles.feedName}>{feed.name}</Text>
                    <Text style={styles.feedProvider}>{feed.provider}</Text>
                  </View>
                  <View style={[styles.statusBadge, { backgroundColor: getStatusColor(feed.status) + '20' }]}>
                    {getStatusIcon(feed.status)}
                    <Text style={[styles.statusText, { color: getStatusColor(feed.status) }]}>
                      {feed.status}
                    </Text>
                  </View>
                </View>
                <View style={styles.feedStats}>
                  <View style={styles.feedStat}>
                    <Text style={styles.feedStatLabel}>Last Sync</Text>
                    <Text style={styles.feedStatValue}>
                      {feed.lastSync.toLocaleTimeString()}
                    </Text>
                  </View>
                  <View style={styles.feedStat}>
                    <Text style={styles.feedStatLabel}>Records</Text>
                    <Text style={styles.feedStatValue}>
                      {feed.recordsProcessed.toLocaleString()}
                    </Text>
                  </View>
                </View>
                <View style={styles.feedActions}>
                  <TouchableOpacity style={styles.feedActionButton}>
                    <Settings size={16} color={Colors.secondary} />
                    <Text style={styles.feedActionText}>Configure</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.feedActionButton}>
                    <Activity size={16} color={Colors.accent} />
                    <Text style={styles.feedActionText}>Test Connection</Text>
                  </TouchableOpacity>
                </View>
              </View>
            ))}
          </View>
        )}

        {selectedTab === 'api' && (
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>API Keys & Integrations</Text>
              <TouchableOpacity style={styles.addButton}>
                <Text style={styles.addButtonText}>+ New Key</Text>
              </TouchableOpacity>
            </View>
            {API_KEYS.map((apiKey) => (
              <View key={apiKey.id} style={styles.apiCard}>
                <View style={styles.apiHeader}>
                  <Key size={20} color={Colors.secondary} />
                  <Text style={styles.apiName}>{apiKey.name}</Text>
                </View>
                <View style={styles.apiKeyContainer}>
                  <TextInput
                    style={styles.apiKeyInput}
                    value={apiKey.key}
                    editable={false}
                    selectTextOnFocus={false}
                  />
                  <TouchableOpacity style={styles.copyButton}>
                    <Text style={styles.copyButtonText}>Copy</Text>
                  </TouchableOpacity>
                </View>
                <View style={styles.apiStats}>
                  <View style={styles.apiStat}>
                    <Text style={styles.apiStatLabel}>Created</Text>
                    <Text style={styles.apiStatValue}>
                      {apiKey.created.toLocaleDateString()}
                    </Text>
                  </View>
                  <View style={styles.apiStat}>
                    <Text style={styles.apiStatLabel}>Last Used</Text>
                    <Text style={styles.apiStatValue}>
                      {apiKey.lastUsed.toLocaleString()}
                    </Text>
                  </View>
                  <View style={styles.apiStat}>
                    <Text style={styles.apiStatLabel}>Total Requests</Text>
                    <Text style={styles.apiStatValue}>
                      {apiKey.requests.toLocaleString()}
                    </Text>
                  </View>
                </View>
                <TouchableOpacity style={styles.revokeButton}>
                  <Text style={styles.revokeButtonText}>Revoke Key</Text>
                </TouchableOpacity>
              </View>
            ))}
          </View>
        )}

        {selectedTab === 'compliance' && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Governance & Compliance Logs</Text>
            <View style={styles.complianceCard}>
              <View style={styles.complianceRow}>
                <Shield size={20} color={Colors.accent} />
                <View style={styles.complianceContent}>
                  <Text style={styles.complianceTitle}>Audit Trail</Text>
                  <Text style={styles.complianceSubtitle}>
                    All system actions logged and encrypted
                  </Text>
                </View>
                <CheckCircle size={20} color={Colors.accent} />
              </View>
              <View style={styles.complianceDivider} />
              <View style={styles.complianceRow}>
                <Shield size={20} color={Colors.accent} />
                <View style={styles.complianceContent}>
                  <Text style={styles.complianceTitle}>Data Encryption</Text>
                  <Text style={styles.complianceSubtitle}>
                    AES-256 encryption at rest and in transit
                  </Text>
                </View>
                <CheckCircle size={20} color={Colors.accent} />
              </View>
              <View style={styles.complianceDivider} />
              <View style={styles.complianceRow}>
                <Shield size={20} color={Colors.accent} />
                <View style={styles.complianceContent}>
                  <Text style={styles.complianceTitle}>Access Control</Text>
                  <Text style={styles.complianceSubtitle}>
                    Role-based permissions enforced
                  </Text>
                </View>
                <CheckCircle size={20} color={Colors.accent} />
              </View>
            </View>
            <TouchableOpacity style={styles.exportButton}>
              <Text style={styles.exportButtonText}>Export Compliance Report</Text>
            </TouchableOpacity>
          </View>
        )}

        {selectedTab === 'analytics' && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Usage Analytics</Text>
            <View style={styles.analyticsGrid}>
              <View style={styles.analyticsCard}>
                <TrendingUp size={24} color={Colors.secondary} />
                <Text style={styles.analyticsValue}>+24%</Text>
                <Text style={styles.analyticsLabel}>API Growth</Text>
              </View>
              <View style={styles.analyticsCard}>
                <Activity size={24} color={Colors.accent} />
                <Text style={styles.analyticsValue}>99.9%</Text>
                <Text style={styles.analyticsLabel}>Uptime</Text>
              </View>
              <View style={styles.analyticsCard}>
                <Users size={24} color={Colors.gold} />
                <Text style={styles.analyticsValue}>1,234</Text>
                <Text style={styles.analyticsLabel}>Active Users</Text>
              </View>
            </View>
            <View style={styles.usageCard}>
              <Text style={styles.usageTitle}>API Usage by Endpoint</Text>
              {[
                { endpoint: '/market-data', calls: 450230, percentage: 45 },
                { endpoint: '/portfolio', calls: 320150, percentage: 32 },
                { endpoint: '/analytics', calls: 180420, percentage: 18 },
                { endpoint: '/auth', calls: 50200, percentage: 5 },
              ].map((item, index) => (
                <View key={index} style={styles.usageRow}>
                  <View style={styles.usageLeft}>
                    <Text style={styles.usageEndpoint}>{item.endpoint}</Text>
                    <View style={styles.usageBar}>
                      <View
                        style={[styles.usageBarFill, { width: `${item.percentage}%` }]}
                      />
                    </View>
                  </View>
                  <Text style={styles.usageCalls}>{item.calls.toLocaleString()}</Text>
                </View>
              ))}
            </View>
          </View>
        )}
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
  statsGrid: {
    flexDirection: 'row',
    paddingHorizontal: 24,
    gap: 12,
    marginBottom: 24,
  },
  statCard: {
    flex: 1,
    backgroundColor: Colors.surface,
    padding: 16,
    borderRadius: 16,
    alignItems: 'center',
    gap: 8,
  },
  statValue: {
    fontSize: 20,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  statLabel: {
    fontSize: 12,
    color: Colors.textSecondary,
    textAlign: 'center',
  },
  tabBar: {
    flexDirection: 'row',
    paddingHorizontal: 24,
    gap: 8,
    marginBottom: 24,
  },
  tab: {
    flex: 1,
    flexDirection: 'column',
    alignItems: 'center',
    gap: 6,
    paddingVertical: 12,
    borderRadius: 12,
    backgroundColor: Colors.surface,
  },
  tabActive: {
    backgroundColor: Colors.secondary + '20',
  },
  tabText: {
    fontSize: 11,
    fontWeight: '600' as const,
    color: Colors.textMuted,
  },
  tabTextActive: {
    color: Colors.secondary,
  },
  section: {
    paddingHorizontal: 24,
    paddingBottom: 24,
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
  },
  addButton: {
    backgroundColor: Colors.secondary,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
  },
  addButtonText: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  feedCard: {
    backgroundColor: Colors.surface,
    padding: 20,
    borderRadius: 16,
    marginBottom: 12,
    gap: 16,
  },
  feedHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  feedLeft: {
    flex: 1,
    gap: 4,
  },
  feedName: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  feedProvider: {
    fontSize: 13,
    color: Colors.textSecondary,
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '600' as const,
  },
  feedStats: {
    flexDirection: 'row',
    gap: 24,
  },
  feedStat: {
    gap: 4,
  },
  feedStatLabel: {
    fontSize: 12,
    color: Colors.textSecondary,
  },
  feedStatValue: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  feedActions: {
    flexDirection: 'row',
    gap: 12,
  },
  feedActionButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 10,
    borderRadius: 8,
    backgroundColor: Colors.surfaceLight,
  },
  feedActionText: {
    fontSize: 13,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  apiCard: {
    backgroundColor: Colors.surface,
    padding: 20,
    borderRadius: 16,
    marginBottom: 12,
    gap: 16,
  },
  apiHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  apiName: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  apiKeyContainer: {
    flexDirection: 'row',
    gap: 12,
  },
  apiKeyInput: {
    flex: 1,
    backgroundColor: Colors.surfaceLight,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 8,
    fontSize: 14,
    color: Colors.text,
    fontFamily: 'monospace',
  },
  copyButton: {
    backgroundColor: Colors.secondary,
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    justifyContent: 'center',
  },
  copyButtonText: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  apiStats: {
    gap: 12,
  },
  apiStat: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  apiStatLabel: {
    fontSize: 13,
    color: Colors.textSecondary,
  },
  apiStatValue: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  revokeButton: {
    paddingVertical: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: Colors.danger,
    alignItems: 'center',
  },
  revokeButtonText: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.danger,
  },
  complianceCard: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    overflow: 'hidden',
    marginBottom: 16,
  },
  complianceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    padding: 16,
  },
  complianceContent: {
    flex: 1,
    gap: 4,
  },
  complianceTitle: {
    fontSize: 15,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  complianceSubtitle: {
    fontSize: 13,
    color: Colors.textSecondary,
  },
  complianceDivider: {
    height: 1,
    backgroundColor: Colors.border,
    marginHorizontal: 16,
  },
  exportButton: {
    backgroundColor: Colors.secondary,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  exportButtonText: {
    fontSize: 15,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  analyticsGrid: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 24,
  },
  analyticsCard: {
    flex: 1,
    backgroundColor: Colors.surface,
    padding: 20,
    borderRadius: 16,
    alignItems: 'center',
    gap: 8,
  },
  analyticsValue: {
    fontSize: 20,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  analyticsLabel: {
    fontSize: 12,
    color: Colors.textSecondary,
    textAlign: 'center',
  },
  usageCard: {
    backgroundColor: Colors.surface,
    padding: 20,
    borderRadius: 16,
    gap: 16,
  },
  usageTitle: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  usageRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  usageLeft: {
    flex: 1,
    gap: 8,
  },
  usageEndpoint: {
    fontSize: 13,
    fontWeight: '600' as const,
    color: Colors.text,
    fontFamily: 'monospace',
  },
  usageBar: {
    height: 6,
    backgroundColor: Colors.surfaceLight,
    borderRadius: 3,
    overflow: 'hidden',
  },
  usageBarFill: {
    height: '100%',
    backgroundColor: Colors.secondary,
    borderRadius: 3,
  },
  usageCalls: {
    fontSize: 13,
    fontWeight: '600' as const,
    color: Colors.textSecondary,
    minWidth: 70,
    textAlign: 'right',
  },
});
