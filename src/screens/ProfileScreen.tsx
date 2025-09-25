import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Switch,
  Alert,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { colors } from '../theme/colors';

interface UserProfile {
  name: string;
  email: string;
  experience: 'beginner' | 'intermediate' | 'advanced';
  riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  investmentGoals: string[];
  timeHorizon: 'short' | 'medium' | 'long';
  subscription: 'free' | 'premium' | 'pro';
  notifications: {
    email: boolean;
    push: boolean;
    sms: boolean;
  };
  privacy: {
    profilePublic: boolean;
    showReturns: boolean;
    showHoldings: boolean;
  };
}

interface ProfileScreenProps {
  onLogout: () => void;
  onBack?: () => void;
}

const ProfileScreen: React.FC<ProfileScreenProps> = ({ onLogout, onBack }) => {
  const [activeTab, setActiveTab] = useState<'profile' | 'settings' | 'billing' | 'security'>('profile');
  const [userProfile, setUserProfile] = useState<UserProfile>({
    name: 'John Doe',
    email: 'john.doe@example.com',
    experience: 'intermediate',
    riskTolerance: 'moderate',
    investmentGoals: ['retirement', 'wealth'],
    timeHorizon: 'long',
    subscription: 'premium',
    notifications: {
      email: true,
      push: true,
      sms: false
    },
    privacy: {
      profilePublic: false,
      showReturns: true,
      showHoldings: false
    }
  });
  
  const [isEditing, setIsEditing] = useState(false);
  const [editForm, setEditForm] = useState({
    name: userProfile.name,
    email: userProfile.email
  });
  
  const isDarkMode = true;
  const c = colors[isDarkMode ? 'dark' : 'light'];

  const handleSaveProfile = () => {
    setUserProfile(prev => ({
      ...prev,
      name: editForm.name,
      email: editForm.email
    }));
    setIsEditing(false);
    Alert.alert('Success', 'Profile updated successfully');
  };

  const handleSubscriptionUpgrade = () => {
    Alert.alert(
      'Upgrade Subscription',
      'Choose your subscription plan:',
      [
        { text: 'Premium - $9.99/month', onPress: () => console.log('Premium selected') },
        { text: 'Pro - $19.99/month', onPress: () => console.log('Pro selected') },
        { text: 'Cancel', style: 'cancel' }
      ]
    );
  };

  const renderProfile = () => (
    <View style={styles.tabContent}>
      <View style={[styles.profileHeader, { backgroundColor: c.surface }]}>
        <View style={styles.avatarContainer}>
          <View style={[styles.avatar, { backgroundColor: c.primary }]}>
            <Text style={[styles.avatarText, { color: c.background }]}>
              {userProfile.name.split(' ').map(n => n[0]).join('')}
            </Text>
          </View>
          <TouchableOpacity style={[styles.editAvatarButton, { backgroundColor: c.primary }]}>
            <Icon name="camera" size={16} color={c.background} />
          </TouchableOpacity>
        </View>
        
        <View style={styles.profileInfo}>
          {isEditing ? (
            <View style={styles.editForm}>
              <TextInput
                style={[styles.editInput, { backgroundColor: c.background, borderColor: c.border, color: c.text }]}
                value={editForm.name}
                onChangeText={(text) => setEditForm(prev => ({ ...prev, name: text }))}
                placeholder="Full Name"
                placeholderTextColor={c.textMuted}
              />
              <TextInput
                style={[styles.editInput, { backgroundColor: c.background, borderColor: c.border, color: c.text }]}
                value={editForm.email}
                onChangeText={(text) => setEditForm(prev => ({ ...prev, email: text }))}
                placeholder="Email"
                placeholderTextColor={c.textMuted}
                keyboardType="email-address"
              />
              <View style={styles.editActions}>
                <TouchableOpacity 
                  style={[styles.cancelButton, { borderColor: c.border }]}
                  onPress={() => setIsEditing(false)}
                >
                  <Text style={[styles.cancelText, { color: c.text }]}>Cancel</Text>
                </TouchableOpacity>
                <TouchableOpacity 
                  style={[styles.saveButton, { backgroundColor: c.primary }]}
                  onPress={handleSaveProfile}
                >
                  <Text style={[styles.saveText, { color: c.background }]}>Save</Text>
                </TouchableOpacity>
              </View>
            </View>
          ) : (
            <>
              <Text style={[styles.profileName, { color: c.text }]}>{userProfile.name}</Text>
              <Text style={[styles.profileEmail, { color: c.textMuted }]}>{userProfile.email}</Text>
              <TouchableOpacity 
                style={[styles.editButton, { backgroundColor: c.primary }]}
                onPress={() => setIsEditing(true)}
              >
                <Icon name="pencil" size={16} color={c.background} />
                <Text style={[styles.editButtonText, { color: c.background }]}>Edit Profile</Text>
              </TouchableOpacity>
            </>
          )}
        </View>
      </View>
      
      <View style={styles.profileStats}>
        <View style={[styles.statCard, { backgroundColor: c.surface }]}>
          <Text style={[styles.statLabel, { color: c.textMuted }]}>Experience Level</Text>
          <Text style={[styles.statValue, { color: c.text }]}>
            {userProfile.experience.charAt(0).toUpperCase() + userProfile.experience.slice(1)}
          </Text>
        </View>
        <View style={[styles.statCard, { backgroundColor: c.surface }]}>
          <Text style={[styles.statLabel, { color: c.textMuted }]}>Risk Tolerance</Text>
          <Text style={[styles.statValue, { color: c.text }]}>
            {userProfile.riskTolerance.charAt(0).toUpperCase() + userProfile.riskTolerance.slice(1)}
          </Text>
        </View>
        <View style={[styles.statCard, { backgroundColor: c.surface }]}>
          <Text style={[styles.statLabel, { color: c.textMuted }]}>Time Horizon</Text>
          <Text style={[styles.statValue, { color: c.text }]}>
            {userProfile.timeHorizon.charAt(0).toUpperCase() + userProfile.timeHorizon.slice(1)} Term
          </Text>
        </View>
      </View>
      
      <View style={[styles.investmentGoals, { backgroundColor: c.surface }]}>
        <Text style={[styles.goalsTitle, { color: c.text }]}>Investment Goals</Text>
        <View style={styles.goalsList}>
          {userProfile.investmentGoals.map((goal, index) => (
            <View key={index} style={[styles.goalTag, { backgroundColor: c.primary + '20' }]}>
              <Text style={[styles.goalText, { color: c.primary }]}>
                {goal.charAt(0).toUpperCase() + goal.slice(1)}
              </Text>
            </View>
          ))}
        </View>
      </View>
    </View>
  );

  const renderSettings = () => (
    <View style={styles.tabContent}>
      <Text style={[styles.sectionTitle, { color: c.text }]}>Settings</Text>
      
      <View style={styles.settingsList}>
        <View style={[styles.settingGroup, { backgroundColor: c.surface }]}>
          <Text style={[styles.groupTitle, { color: c.text }]}>Notifications</Text>
          
          <View style={styles.settingItem}>
            <View style={styles.settingInfo}>
              <Text style={[styles.settingName, { color: c.text }]}>Email Notifications</Text>
              <Text style={[styles.settingDescription, { color: c.textMuted }]}>
                Receive updates via email
              </Text>
            </View>
            <Switch
              value={userProfile.notifications.email}
              onValueChange={(value) => setUserProfile(prev => ({
                ...prev,
                notifications: { ...prev.notifications, email: value }
              }))}
              trackColor={{ false: c.border, true: c.primary }}
              thumbColor={userProfile.notifications.email ? c.background : c.textMuted}
            />
          </View>
          
          <View style={styles.settingItem}>
            <View style={styles.settingInfo}>
              <Text style={[styles.settingName, { color: c.text }]}>Push Notifications</Text>
              <Text style={[styles.settingDescription, { color: c.textMuted }]}>
                Receive push notifications
              </Text>
            </View>
            <Switch
              value={userProfile.notifications.push}
              onValueChange={(value) => setUserProfile(prev => ({
                ...prev,
                notifications: { ...prev.notifications, push: value }
              }))}
              trackColor={{ false: c.border, true: c.primary }}
              thumbColor={userProfile.notifications.push ? c.background : c.textMuted}
            />
          </View>
          
          <View style={styles.settingItem}>
            <View style={styles.settingInfo}>
              <Text style={[styles.settingName, { color: c.text }]}>SMS Alerts</Text>
              <Text style={[styles.settingDescription, { color: c.textMuted }]}>
                Receive critical alerts via SMS
              </Text>
            </View>
            <Switch
              value={userProfile.notifications.sms}
              onValueChange={(value) => setUserProfile(prev => ({
                ...prev,
                notifications: { ...prev.notifications, sms: value }
              }))}
              trackColor={{ false: c.border, true: c.primary }}
              thumbColor={userProfile.notifications.sms ? c.background : c.textMuted}
            />
          </View>
        </View>
        
        <View style={[styles.settingGroup, { backgroundColor: c.surface }]}>
          <Text style={[styles.groupTitle, { color: c.text }]}>Privacy</Text>
          
          <View style={styles.settingItem}>
            <View style={styles.settingInfo}>
              <Text style={[styles.settingName, { color: c.text }]}>Public Profile</Text>
              <Text style={[styles.settingDescription, { color: c.textMuted }]}>
                Make your profile visible to other users
              </Text>
            </View>
            <Switch
              value={userProfile.privacy.profilePublic}
              onValueChange={(value) => setUserProfile(prev => ({
                ...prev,
                privacy: { ...prev.privacy, profilePublic: value }
              }))}
              trackColor={{ false: c.border, true: c.primary }}
              thumbColor={userProfile.privacy.profilePublic ? c.background : c.textMuted}
            />
          </View>
          
          <View style={styles.settingItem}>
            <View style={styles.settingInfo}>
              <Text style={[styles.settingName, { color: c.text }]}>Show Returns</Text>
              <Text style={[styles.settingDescription, { color: c.textMuted }]}>
                Display your returns publicly
              </Text>
            </View>
            <Switch
              value={userProfile.privacy.showReturns}
              onValueChange={(value) => setUserProfile(prev => ({
                ...prev,
                privacy: { ...prev.privacy, showReturns: value }
              }))}
              trackColor={{ false: c.border, true: c.primary }}
              thumbColor={userProfile.privacy.showReturns ? c.background : c.textMuted}
            />
          </View>
        </View>
      </View>
    </View>
  );

  const renderBilling = () => (
    <View style={styles.tabContent}>
      <Text style={[styles.sectionTitle, { color: c.text }]}>Billing & Subscription</Text>
      
      <View style={[styles.subscriptionCard, { backgroundColor: c.surface }]}>
        <View style={styles.subscriptionHeader}>
          <Text style={[styles.subscriptionTitle, { color: c.text }]}>Current Plan</Text>
          <View style={[styles.subscriptionBadge, { backgroundColor: c.primary }]}>
            <Text style={[styles.subscriptionBadgeText, { color: c.background }]}>
              {userProfile.subscription.toUpperCase()}
            </Text>
          </View>
        </View>
        
        <Text style={[styles.subscriptionDescription, { color: c.textMuted }]}>
          {userProfile.subscription === 'free' 
            ? 'Basic features with limited access'
            : userProfile.subscription === 'premium'
            ? 'Advanced analytics and priority support'
            : 'Full platform access with premium features'
          }
        </Text>
        
        {userProfile.subscription === 'free' && (
          <TouchableOpacity 
            style={[styles.upgradeButton, { backgroundColor: c.primary }]}
            onPress={handleSubscriptionUpgrade}
          >
            <Text style={[styles.upgradeText, { color: c.background }]}>Upgrade Now</Text>
          </TouchableOpacity>
        )}
      </View>
      
      <View style={[styles.billingHistory, { backgroundColor: c.surface }]}>
        <Text style={[styles.billingTitle, { color: c.text }]}>Billing History</Text>
        <View style={styles.billingItem}>
          <Text style={[styles.billingDate, { color: c.textMuted }]}>January 2024</Text>
          <Text style={[styles.billingAmount, { color: c.text }]}>$9.99</Text>
        </View>
        <View style={styles.billingItem}>
          <Text style={[styles.billingDate, { color: c.textMuted }]}>December 2023</Text>
          <Text style={[styles.billingAmount, { color: c.text }]}>$9.99</Text>
        </View>
      </View>
    </View>
  );

  const renderSecurity = () => (
    <View style={styles.tabContent}>
      <Text style={[styles.sectionTitle, { color: c.text }]}>Security</Text>
      
      <View style={styles.securityList}>
        <TouchableOpacity style={[styles.securityItem, { backgroundColor: c.surface }]}>
          <Icon name="lock" size={24} color={c.primary} />
          <View style={styles.securityInfo}>
            <Text style={[styles.securityName, { color: c.text }]}>Change Password</Text>
            <Text style={[styles.securityDescription, { color: c.textMuted }]}>
              Update your account password
            </Text>
          </View>
          <Icon name="chevron-right" size={20} color={c.textMuted} />
        </TouchableOpacity>
        
        <TouchableOpacity style={[styles.securityItem, { backgroundColor: c.surface }]}>
          <Icon name="shield-check" size={24} color={c.success} />
          <View style={styles.securityInfo}>
            <Text style={[styles.securityName, { color: c.text }]}>Two-Factor Authentication</Text>
            <Text style={[styles.securityDescription, { color: c.textMuted }]}>
              Add an extra layer of security
            </Text>
          </View>
          <Switch
            value={true}
            trackColor={{ false: c.border, true: c.success }}
            thumbColor={c.background}
          />
        </TouchableOpacity>
        
        <TouchableOpacity style={[styles.securityItem, { backgroundColor: c.surface }]}>
          <Icon name="key" size={24} color={c.warning} />
          <View style={styles.securityInfo}>
            <Text style={[styles.securityName, { color: c.text }]}>API Keys</Text>
            <Text style={[styles.securityDescription, { color: c.textMuted }]}>
              Manage your API access keys
            </Text>
          </View>
          <Icon name="chevron-right" size={20} color={c.textMuted} />
        </TouchableOpacity>
        
        <TouchableOpacity style={[styles.securityItem, { backgroundColor: c.surface }]}>
          <Icon name="logout" size={24} color={c.danger} />
          <View style={styles.securityInfo}>
            <Text style={[styles.securityName, { color: c.danger }]}>Sign Out</Text>
            <Text style={[styles.securityDescription, { color: c.textMuted }]}>
              Sign out of your account
            </Text>
          </View>
          <Icon name="chevron-right" size={20} color={c.textMuted} />
        </TouchableOpacity>
      </View>
    </View>
  );

  return (
    <View style={[styles.container, { backgroundColor: c.background }]}>
      {/* Header */}
      <View style={[styles.header, { borderBottomColor: c.border }]}>
        {onBack && (
          <TouchableOpacity onPress={onBack} style={styles.backButton}>
            <Icon name="arrow-left" size={24} color={c.text} />
          </TouchableOpacity>
        )}
        <Text style={[styles.headerTitle, { color: c.text }]}>Profile & Settings</Text>
        <View style={styles.headerActions}>
          <TouchableOpacity style={styles.headerAction} onPress={onLogout}>
            <Icon name="logout" size={24} color={c.danger} />
          </TouchableOpacity>
        </View>
      </View>

      {/* Tabs */}
      <View style={[styles.tabsContainer, { backgroundColor: c.surface }]}>
        {[
          { id: 'profile', label: 'Profile', icon: 'account' },
          { id: 'settings', label: 'Settings', icon: 'cog' },
          { id: 'billing', label: 'Billing', icon: 'credit-card' },
          { id: 'security', label: 'Security', icon: 'shield' }
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
        {activeTab === 'profile' && renderProfile()}
        {activeTab === 'settings' && renderSettings()}
        {activeTab === 'billing' && renderBilling()}
        {activeTab === 'security' && renderSecurity()}
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
  profileHeader: {
    padding: 20,
    borderRadius: 12,
    marginBottom: 20,
  },
  avatarContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    fontSize: 24,
    fontWeight: '700',
  },
  editAvatarButton: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 28,
    height: 28,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  profileInfo: {
    alignItems: 'center',
  },
  profileName: {
    fontSize: 24,
    fontWeight: '700',
    marginBottom: 4,
  },
  profileEmail: {
    fontSize: 16,
    marginBottom: 16,
  },
  editButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
    gap: 8,
  },
  editButtonText: {
    fontSize: 14,
    fontWeight: '600',
  },
  editForm: {
    width: '100%',
    gap: 16,
  },
  editInput: {
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 16,
  },
  editActions: {
    flexDirection: 'row',
    gap: 12,
  },
  cancelButton: {
    flex: 1,
    borderWidth: 1,
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  cancelText: {
    fontSize: 16,
    fontWeight: '600',
  },
  saveButton: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  saveText: {
    fontSize: 16,
    fontWeight: '600',
  },
  profileStats: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 20,
  },
  statCard: {
    flex: 1,
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  statLabel: {
    fontSize: 12,
    marginBottom: 8,
  },
  statValue: {
    fontSize: 16,
    fontWeight: '700',
  },
  investmentGoals: {
    padding: 16,
    borderRadius: 12,
  },
  goalsTitle: {
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 12,
  },
  goalsList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  goalTag: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  goalText: {
    fontSize: 14,
    fontWeight: '600',
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '700',
    marginBottom: 20,
  },
  settingsList: {
    gap: 16,
  },
  settingGroup: {
    padding: 16,
    borderRadius: 12,
  },
  groupTitle: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 16,
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
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
  subscriptionCard: {
    padding: 20,
    borderRadius: 12,
    marginBottom: 20,
  },
  subscriptionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  subscriptionTitle: {
    fontSize: 18,
    fontWeight: '700',
  },
  subscriptionBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  subscriptionBadgeText: {
    fontSize: 12,
    fontWeight: '700',
  },
  subscriptionDescription: {
    fontSize: 14,
    marginBottom: 16,
  },
  upgradeButton: {
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  upgradeText: {
    fontSize: 16,
    fontWeight: '700',
  },
  billingHistory: {
    padding: 16,
    borderRadius: 12,
  },
  billingTitle: {
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 12,
  },
  billingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 8,
  },
  billingDate: {
    fontSize: 14,
  },
  billingAmount: {
    fontSize: 14,
    fontWeight: '600',
  },
  securityList: {
    gap: 12,
  },
  securityItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 12,
    gap: 16,
  },
  securityInfo: {
    flex: 1,
  },
  securityName: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  securityDescription: {
    fontSize: 14,
    lineHeight: 20,
  },
});

export default ProfileScreen;