import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Switch,
} from 'react-native';
import { Stack } from 'expo-router';
import {
  Bell,
  TrendingUp,
  AlertTriangle,
  Trophy,
  Settings,
  Clock,
  CheckCircle,
  XCircle,
} from 'lucide-react-native';
import Colors from '@/constants/colors';

interface Notification {
  id: string;
  type: 'market' | 'portfolio' | 'competition' | 'system';
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
  priority: 'high' | 'medium' | 'low';
}

const NOTIFICATIONS: Notification[] = [
  {
    id: '1',
    type: 'portfolio',
    title: 'Rebalancing Suggested',
    message: 'Your tech allocation is 5% over target. Consider rebalancing.',
    timestamp: new Date('2025-01-15T10:30:00'),
    read: false,
    priority: 'high',
  },
  {
    id: '2',
    type: 'market',
    title: 'Market Alert',
    message: 'S&P 500 dropped 2% in the last hour. Review your positions.',
    timestamp: new Date('2025-01-15T09:15:00'),
    read: false,
    priority: 'high',
  },
  {
    id: '3',
    type: 'competition',
    title: 'New Tournament Starting',
    message: '2008 Financial Crisis scenario begins in 1 hour.',
    timestamp: new Date('2025-01-15T08:00:00'),
    read: true,
    priority: 'medium',
  },
  {
    id: '4',
    type: 'system',
    title: 'Model Update',
    message: 'AI model updated to v2.4.1 with improved accuracy.',
    timestamp: new Date('2025-01-14T16:00:00'),
    read: true,
    priority: 'low',
  },
  {
    id: '5',
    type: 'portfolio',
    title: 'Dividend Received',
    message: 'AAPL dividend of $24.50 credited to your account.',
    timestamp: new Date('2025-01-14T12:00:00'),
    read: true,
    priority: 'low',
  },
  {
    id: '6',
    type: 'competition',
    title: 'Leaderboard Update',
    message: 'You moved up to #142 in the global rankings!',
    timestamp: new Date('2025-01-13T18:30:00'),
    read: true,
    priority: 'medium',
  },
];

export default function NotificationsScreen() {
  const [notifications, setNotifications] = useState(NOTIFICATIONS);
  const [filter, setFilter] = useState<'all' | 'unread'>('all');
  const [settings, setSettings] = useState({
    marketAlerts: true,
    portfolioUpdates: true,
    competitionNews: true,
    systemAlerts: false,
  });

  const filteredNotifications = notifications.filter((n) =>
    filter === 'all' ? true : !n.read
  );

  const markAsRead = (id: string) => {
    setNotifications(
      notifications.map((n) => (n.id === id ? { ...n, read: true } : n))
    );
  };

  const markAllAsRead = () => {
    setNotifications(notifications.map((n) => ({ ...n, read: true })));
  };

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'market':
        return <TrendingUp size={20} color={Colors.secondary} />;
      case 'portfolio':
        return <AlertTriangle size={20} color={Colors.warning} />;
      case 'competition':
        return <Trophy size={20} color={Colors.gold} />;
      case 'system':
        return <Settings size={20} color={Colors.textMuted} />;
      default:
        return <Bell size={20} color={Colors.textMuted} />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return Colors.danger;
      case 'medium':
        return Colors.warning;
      case 'low':
        return Colors.textMuted;
      default:
        return Colors.textMuted;
    }
  };

  return (
    <View style={styles.container}>
      <Stack.Screen
        options={{
          title: 'Notifications',
          headerStyle: { backgroundColor: Colors.background },
          headerTintColor: Colors.text,
        }}
      />
      <ScrollView showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <View style={styles.filterButtons}>
            <TouchableOpacity
              style={[styles.filterButton, filter === 'all' && styles.filterButtonActive]}
              onPress={() => setFilter('all')}
            >
              <Text
                style={[
                  styles.filterButtonText,
                  filter === 'all' && styles.filterButtonTextActive,
                ]}
              >
                All
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.filterButton, filter === 'unread' && styles.filterButtonActive]}
              onPress={() => setFilter('unread')}
            >
              <Text
                style={[
                  styles.filterButtonText,
                  filter === 'unread' && styles.filterButtonTextActive,
                ]}
              >
                Unread ({notifications.filter((n) => !n.read).length})
              </Text>
            </TouchableOpacity>
          </View>
          {notifications.some((n) => !n.read) && (
            <TouchableOpacity style={styles.markAllButton} onPress={markAllAsRead}>
              <CheckCircle size={16} color={Colors.secondary} />
              <Text style={styles.markAllText}>Mark all read</Text>
            </TouchableOpacity>
          )}
        </View>

        <View style={styles.section}>
          {filteredNotifications.length === 0 ? (
            <View style={styles.emptyState}>
              <Bell size={48} color={Colors.textMuted} />
              <Text style={styles.emptyTitle}>No notifications</Text>
              <Text style={styles.emptySubtitle}>
                {filter === 'unread'
                  ? "You're all caught up!"
                  : 'Notifications will appear here'}
              </Text>
            </View>
          ) : (
            filteredNotifications.map((notification) => (
              <TouchableOpacity
                key={notification.id}
                style={[
                  styles.notificationCard,
                  !notification.read && styles.notificationCardUnread,
                ]}
                onPress={() => markAsRead(notification.id)}
              >
                <View style={styles.notificationLeft}>
                  <View
                    style={[
                      styles.notificationIcon,
                      {
                        backgroundColor:
                          notification.type === 'market'
                            ? Colors.secondary + '20'
                            : notification.type === 'portfolio'
                            ? Colors.warning + '20'
                            : notification.type === 'competition'
                            ? Colors.gold + '20'
                            : Colors.surfaceLight,
                      },
                    ]}
                  >
                    {getNotificationIcon(notification.type)}
                  </View>
                  <View style={styles.notificationContent}>
                    <View style={styles.notificationHeader}>
                      <Text style={styles.notificationTitle}>{notification.title}</Text>
                      {!notification.read && <View style={styles.unreadDot} />}
                    </View>
                    <Text style={styles.notificationMessage}>{notification.message}</Text>
                    <View style={styles.notificationFooter}>
                      <Clock size={12} color={Colors.textMuted} />
                      <Text style={styles.notificationTime}>
                        {notification.timestamp.toLocaleString()}
                      </Text>
                      <View
                        style={[
                          styles.priorityDot,
                          { backgroundColor: getPriorityColor(notification.priority) },
                        ]}
                      />
                    </View>
                  </View>
                </View>
              </TouchableOpacity>
            ))
          )}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Notification Settings</Text>
          <View style={styles.settingsCard}>
            <View style={styles.settingRow}>
              <View style={styles.settingLeft}>
                <TrendingUp size={20} color={Colors.secondary} />
                <View style={styles.settingContent}>
                  <Text style={styles.settingTitle}>Market Events</Text>
                  <Text style={styles.settingSubtitle}>Price alerts and market news</Text>
                </View>
              </View>
              <Switch
                value={settings.marketAlerts}
                onValueChange={(value) =>
                  setSettings({ ...settings, marketAlerts: value })
                }
                trackColor={{ false: Colors.surfaceLight, true: Colors.secondary + '60' }}
                thumbColor={settings.marketAlerts ? Colors.secondary : Colors.textMuted}
              />
            </View>

            <View style={styles.settingDivider} />

            <View style={styles.settingRow}>
              <View style={styles.settingLeft}>
                <AlertTriangle size={20} color={Colors.warning} />
                <View style={styles.settingContent}>
                  <Text style={styles.settingTitle}>Portfolio Updates</Text>
                  <Text style={styles.settingSubtitle}>Rebalancing and performance</Text>
                </View>
              </View>
              <Switch
                value={settings.portfolioUpdates}
                onValueChange={(value) =>
                  setSettings({ ...settings, portfolioUpdates: value })
                }
                trackColor={{ false: Colors.surfaceLight, true: Colors.secondary + '60' }}
                thumbColor={settings.portfolioUpdates ? Colors.secondary : Colors.textMuted}
              />
            </View>

            <View style={styles.settingDivider} />

            <View style={styles.settingRow}>
              <View style={styles.settingLeft}>
                <Trophy size={20} color={Colors.gold} />
                <View style={styles.settingContent}>
                  <Text style={styles.settingTitle}>Competition Updates</Text>
                  <Text style={styles.settingSubtitle}>Tournaments and rankings</Text>
                </View>
              </View>
              <Switch
                value={settings.competitionNews}
                onValueChange={(value) =>
                  setSettings({ ...settings, competitionNews: value })
                }
                trackColor={{ false: Colors.surfaceLight, true: Colors.secondary + '60' }}
                thumbColor={settings.competitionNews ? Colors.secondary : Colors.textMuted}
              />
            </View>

            <View style={styles.settingDivider} />

            <View style={styles.settingRow}>
              <View style={styles.settingLeft}>
                <Settings size={20} color={Colors.textMuted} />
                <View style={styles.settingContent}>
                  <Text style={styles.settingTitle}>System Alerts</Text>
                  <Text style={styles.settingSubtitle}>Model updates and maintenance</Text>
                </View>
              </View>
              <Switch
                value={settings.systemAlerts}
                onValueChange={(value) =>
                  setSettings({ ...settings, systemAlerts: value })
                }
                trackColor={{ false: Colors.surfaceLight, true: Colors.secondary + '60' }}
                thumbColor={settings.systemAlerts ? Colors.secondary : Colors.textMuted}
              />
            </View>
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
    gap: 16,
  },
  filterButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  filterButton: {
    flex: 1,
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 12,
    backgroundColor: Colors.surface,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  filterButtonActive: {
    borderColor: Colors.secondary,
    backgroundColor: Colors.secondary + '20',
  },
  filterButtonText: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.textSecondary,
  },
  filterButtonTextActive: {
    color: Colors.secondary,
  },
  markAllButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 10,
  },
  markAllText: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.secondary,
  },
  section: {
    paddingHorizontal: 24,
    paddingBottom: 24,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '700' as const,
    color: Colors.text,
    marginBottom: 16,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
    gap: 12,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  emptySubtitle: {
    fontSize: 14,
    color: Colors.textSecondary,
  },
  notificationCard: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
  },
  notificationCardUnread: {
    borderWidth: 2,
    borderColor: Colors.secondary + '40',
  },
  notificationLeft: {
    flexDirection: 'row',
    gap: 12,
  },
  notificationIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  notificationContent: {
    flex: 1,
    gap: 6,
  },
  notificationHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  notificationTitle: {
    fontSize: 15,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  unreadDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: Colors.secondary,
  },
  notificationMessage: {
    fontSize: 14,
    color: Colors.textSecondary,
    lineHeight: 20,
  },
  notificationFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  notificationTime: {
    fontSize: 12,
    color: Colors.textMuted,
  },
  priorityDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginLeft: 6,
  },
  settingsCard: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    overflow: 'hidden',
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
    flex: 1,
  },
  settingContent: {
    flex: 1,
    gap: 4,
  },
  settingTitle: {
    fontSize: 15,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  settingSubtitle: {
    fontSize: 13,
    color: Colors.textSecondary,
  },
  settingDivider: {
    height: 1,
    backgroundColor: Colors.border,
    marginHorizontal: 16,
  },
});
