import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Switch,
  Alert,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { colors } from '../theme/colors';

interface Notification {
  id: string;
  title: string;
  message: string;
  type: 'market' | 'portfolio' | 'system' | 'achievement';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  timestamp: string;
  isRead: boolean;
  action?: string;
}

interface AlertRule {
  id: string;
  name: string;
  condition: string;
  threshold: number;
  isActive: boolean;
  frequency: 'immediate' | 'daily' | 'weekly';
}

interface NotificationsScreenProps {
  onBack: () => void;
}

const NotificationsScreen: React.FC<NotificationsScreenProps> = ({ onBack }) => {
  const [activeTab, setActiveTab] = useState<'notifications' | 'alerts' | 'settings'>('notifications');
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [alertRules, setAlertRules] = useState<AlertRule[]>([]);
  const [notificationSettings, setNotificationSettings] = useState({
    marketUpdates: true,
    portfolioAlerts: true,
    systemNotifications: true,
    achievements: true,
    pushNotifications: true,
    emailNotifications: false,
    smsAlerts: false
  });
  
  const isDarkMode = true;
  const c = colors[isDarkMode ? 'dark' : 'light'];

  useEffect(() => {
    // Load sample notifications
    const sampleNotifications: Notification[] = [
      {
        id: '1',
        title: 'Market Alert',
        message: 'S&P 500 has dropped 2.5% in the last hour',
        type: 'market',
        priority: 'high',
        timestamp: '2024-01-15T10:30:00Z',
        isRead: false,
        action: 'View Market Data'
      },
      {
        id: '2',
        title: 'Portfolio Rebalancing',
        message: 'Your portfolio allocation has drifted from target. Consider rebalancing.',
        type: 'portfolio',
        priority: 'medium',
        timestamp: '2024-01-15T09:15:00Z',
        isRead: false,
        action: 'Rebalance Portfolio'
      },
      {
        id: '3',
        title: 'Achievement Unlocked',
        message: 'Congratulations! You\'ve completed your first month of investing.',
        type: 'achievement',
        priority: 'low',
        timestamp: '2024-01-15T08:00:00Z',
        isRead: true
      },
      {
        id: '4',
        title: 'System Maintenance',
        message: 'Scheduled maintenance will occur tonight from 2-4 AM EST.',
        type: 'system',
        priority: 'medium',
        timestamp: '2024-01-14T16:30:00Z',
        isRead: true
      },
      {
        id: '5',
        title: 'Price Alert',
        message: 'AAPL has reached your target price of $175.00',
        type: 'market',
        priority: 'high',
        timestamp: '2024-01-14T14:22:00Z',
        isRead: true,
        action: 'View Position'
      }
    ];

    const sampleAlertRules: AlertRule[] = [
      {
        id: '1',
        name: 'Portfolio Drop Alert',
        condition: 'Portfolio value drops by',
        threshold: 5,
        isActive: true,
        frequency: 'immediate'
      },
      {
        id: '2',
        name: 'Stock Price Alert',
        condition: 'Stock price changes by',
        threshold: 10,
        isActive: true,
        frequency: 'daily'
      },
      {
        id: '3',
        name: 'Market Volatility Alert',
        condition: 'Market volatility exceeds',
        threshold: 20,
        isActive: false,
        frequency: 'weekly'
      }
    ];

    setNotifications(sampleNotifications);
    setAlertRules(sampleAlertRules);
  }, []);

  const markAsRead = (notificationId: string) => {
    setNotifications(prev => 
      prev.map(notif => 
        notif.id === notificationId ? { ...notif, isRead: true } : notif
      )
    );
  };

  const markAllAsRead = () => {
    setNotifications(prev => 
      prev.map(notif => ({ ...notif, isRead: true }))
    );
  };

  const deleteNotification = (notificationId: string) => {
    setNotifications(prev => prev.filter(notif => notif.id !== notificationId));
  };

  const toggleAlertRule = (ruleId: string) => {
    setAlertRules(prev => 
      prev.map(rule => 
        rule.id === ruleId ? { ...rule, isActive: !rule.isActive } : rule
      )
    );
  };

  const handleNotificationAction = (notification: Notification) => {
    if (notification.action) {
      Alert.alert(
        notification.title,
        `Action: ${notification.action}`,
        [{ text: 'OK' }]
      );
    }
    markAsRead(notification.id);
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'urgent': return c.danger;
      case 'high': return c.warning;
      case 'medium': return c.primary;
      case 'low': return c.textMuted;
      default: return c.textMuted;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'market': return 'chart-line';
      case 'portfolio': return 'briefcase';
      case 'system': return 'cog';
      case 'achievement': return 'trophy';
      default: return 'bell';
    }
  };

  const renderNotifications = () => (
    <View style={styles.tabContent}>
      <View style={styles.notificationsHeader}>
        <Text style={[styles.sectionTitle, { color: c.text }]}>Notifications</Text>
        <TouchableOpacity onPress={markAllAsRead}>
          <Text style={[styles.markAllText, { color: c.primary }]}>Mark All Read</Text>
        </TouchableOpacity>
      </View>
      
      <View style={styles.notificationsList}>
        {notifications.map((notification) => (
          <TouchableOpacity
            key={notification.id}
            style={[
              styles.notificationCard,
              { backgroundColor: c.surface },
              !notification.isRead && { borderLeftColor: c.primary, borderLeftWidth: 4 }
            ]}
            onPress={() => handleNotificationAction(notification)}
          >
            <View style={styles.notificationHeader}>
              <View style={styles.notificationInfo}>
                <View style={styles.notificationTitleRow}>
                  <Icon 
                    name={getTypeIcon(notification.type)} 
                    size={20} 
                    color={getPriorityColor(notification.priority)} 
                  />
                  <Text style={[styles.notificationTitle, { color: c.text }]}>
                    {notification.title}
                  </Text>
                  {!notification.isRead && (
                    <View style={[styles.unreadDot, { backgroundColor: c.primary }]} />
                  )}
                </View>
                <Text style={[styles.notificationMessage, { color: c.textMuted }]}>
                  {notification.message}
                </Text>
                <Text style={[styles.notificationTime, { color: c.textMuted }]}>
                  {new Date(notification.timestamp).toLocaleString()}
                </Text>
              </View>
              
              <View style={styles.notificationActions}>
                <TouchableOpacity
                  onPress={() => deleteNotification(notification.id)}
                  style={styles.deleteButton}
                >
                  <Icon name="delete" size={16} color={c.danger} />
                </TouchableOpacity>
              </View>
            </View>
            
            {notification.action && (
              <View style={styles.actionContainer}>
                <TouchableOpacity style={[styles.actionButton, { backgroundColor: c.primary }]}>
                  <Text style={[styles.actionText, { color: c.background }]}>
                    {notification.action}
                  </Text>
                  <Icon name="arrow-right" size={16} color={c.background} />
                </TouchableOpacity>
              </View>
            )}
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );

  const renderAlerts = () => (
    <View style={styles.tabContent}>
      <Text style={[styles.sectionTitle, { color: c.text }]}>Alert Rules</Text>
      
      <View style={styles.alertRulesList}>
        {alertRules.map((rule) => (
          <View key={rule.id} style={[styles.alertRuleCard, { backgroundColor: c.surface }]}>
            <View style={styles.alertRuleHeader}>
              <Text style={[styles.alertRuleName, { color: c.text }]}>{rule.name}</Text>
              <Switch
                value={rule.isActive}
                onValueChange={() => toggleAlertRule(rule.id)}
                trackColor={{ false: c.border, true: c.primary }}
                thumbColor={rule.isActive ? c.background : c.textMuted}
              />
            </View>
            
            <Text style={[styles.alertRuleCondition, { color: c.textMuted }]}>
              {rule.condition} {rule.threshold}%
            </Text>
            
            <View style={styles.alertRuleDetails}>
              <View style={styles.alertRuleDetail}>
                <Text style={[styles.alertRuleLabel, { color: c.textMuted }]}>Frequency</Text>
                <Text style={[styles.alertRuleValue, { color: c.text }]}>{rule.frequency}</Text>
              </View>
              <View style={styles.alertRuleDetail}>
                <Text style={[styles.alertRuleLabel, { color: c.textMuted }]}>Status</Text>
                <Text style={[
                  styles.alertRuleStatus,
                  { color: rule.isActive ? c.success : c.textMuted }
                ]}>
                  {rule.isActive ? 'Active' : 'Inactive'}
                </Text>
              </View>
            </View>
          </View>
        ))}
      </View>
      
      <TouchableOpacity style={[styles.addAlertButton, { backgroundColor: c.primary }]}>
        <Icon name="plus" size={20} color={c.background} />
        <Text style={[styles.addAlertText, { color: c.background }]}>Add New Alert</Text>
      </TouchableOpacity>
    </View>
  );

  const renderSettings = () => (
    <View style={styles.tabContent}>
      <Text style={[styles.sectionTitle, { color: c.text }]}>Notification Settings</Text>
      
      <View style={styles.settingsList}>
        {Object.entries(notificationSettings).map(([key, value]) => (
          <View key={key} style={[styles.settingItem, { backgroundColor: c.surface }]}>
            <View style={styles.settingInfo}>
              <Text style={[styles.settingName, { color: c.text }]}>
                {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
              </Text>
              <Text style={[styles.settingDescription, { color: c.textMuted }]}>
                {getSettingDescription(key)}
              </Text>
            </View>
            <Switch
              value={value}
              onValueChange={(newValue) => 
                setNotificationSettings(prev => ({ ...prev, [key]: newValue }))
              }
              trackColor={{ false: c.border, true: c.primary }}
              thumbColor={value ? c.background : c.textMuted}
            />
          </View>
        ))}
      </View>
    </View>
  );

  const getSettingDescription = (key: string) => {
    const descriptions: Record<string, string> = {
      marketUpdates: 'Receive alerts about market movements and news',
      portfolioAlerts: 'Get notified about portfolio performance and rebalancing',
      systemNotifications: 'System updates and maintenance notifications',
      achievements: 'Celebrate your investing milestones and progress',
      pushNotifications: 'Receive push notifications on your device',
      emailNotifications: 'Get notifications via email',
      smsAlerts: 'Receive critical alerts via SMS'
    };
    return descriptions[key] || '';
  };

  return (
    <View style={[styles.container, { backgroundColor: c.background }]}>
      {/* Header */}
      <View style={[styles.header, { borderBottomColor: c.border }]}>
        <TouchableOpacity onPress={onBack} style={styles.backButton}>
          <Icon name="arrow-left" size={24} color={c.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: c.text }]}>Notifications & Alerts</Text>
        <View style={styles.headerActions}>
          <TouchableOpacity style={styles.headerAction}>
            <Icon name="cog" size={24} color={c.textMuted} />
          </TouchableOpacity>
        </View>
      </View>

      {/* Tabs */}
      <View style={[styles.tabsContainer, { backgroundColor: c.surface }]}>
        {[
          { id: 'notifications', label: 'Notifications', icon: 'bell' },
          { id: 'alerts', label: 'Alerts', icon: 'alert' },
          { id: 'settings', label: 'Settings', icon: 'cog' }
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
        {activeTab === 'notifications' && renderNotifications()}
        {activeTab === 'alerts' && renderAlerts()}
        {activeTab === 'settings' && renderSettings()}
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
  notificationsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '700',
  },
  markAllText: {
    fontSize: 14,
    fontWeight: '600',
  },
  notificationsList: {
    gap: 12,
  },
  notificationCard: {
    padding: 16,
    borderRadius: 12,
  },
  notificationHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
  },
  notificationInfo: {
    flex: 1,
  },
  notificationTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  notificationTitle: {
    fontSize: 16,
    fontWeight: '700',
    flex: 1,
  },
  unreadDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  notificationMessage: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 8,
  },
  notificationTime: {
    fontSize: 12,
  },
  notificationActions: {
    marginLeft: 12,
  },
  deleteButton: {
    padding: 8,
  },
  actionContainer: {
    marginTop: 12,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    gap: 8,
  },
  actionText: {
    fontSize: 14,
    fontWeight: '600',
  },
  alertRulesList: {
    gap: 16,
    marginBottom: 20,
  },
  alertRuleCard: {
    padding: 16,
    borderRadius: 12,
  },
  alertRuleHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  alertRuleName: {
    fontSize: 16,
    fontWeight: '700',
    flex: 1,
  },
  alertRuleCondition: {
    fontSize: 14,
    marginBottom: 12,
  },
  alertRuleDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  alertRuleDetail: {
    alignItems: 'center',
  },
  alertRuleLabel: {
    fontSize: 12,
    marginBottom: 4,
  },
  alertRuleValue: {
    fontSize: 14,
    fontWeight: '600',
  },
  alertRuleStatus: {
    fontSize: 14,
    fontWeight: '600',
  },
  addAlertButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    borderRadius: 12,
    gap: 8,
  },
  addAlertText: {
    fontSize: 16,
    fontWeight: '700',
  },
  settingsList: {
    gap: 12,
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    borderRadius: 12,
  },
  settingInfo: {
    flex: 1,
  },
  settingName: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  settingDescription: {
    fontSize: 14,
    lineHeight: 20,
  },
});

export default NotificationsScreen;
