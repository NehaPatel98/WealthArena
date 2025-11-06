/**
 * WealthArena API Service - Complete Integration
 * Comprehensive API service for all backend endpoints
 */

import { API_CONFIG } from '../config/apiConfig';
import AsyncStorage from '@react-native-async-storage/async-storage';

const API_BASE_URL = API_CONFIG.BACKEND_BASE_URL;
const CHATBOT_URL = API_CONFIG.CHATBOT_BASE_URL;

// Helper function to get auth headers
const getAuthHeaders = async () => {
  const token = await AsyncStorage.getItem('authToken');
  return {
    'Content-Type': 'application/json',
    ...(token && { 'Authorization': `Bearer ${token}` }),
  };
};

// Helper function to handle API responses
const handleResponse = async (response: Response) => {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
  }
  return response.json();
};

export const apiService = {
  // Authentication
  async signup(userData: {
    email: string;
    password: string;
    username: string;
    firstName?: string;
    lastName?: string;
    displayName?: string;
    full_name?: string;
  }) {
    const response = await fetch(`${API_BASE_URL}/api/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(userData)
    });
    return handleResponse(response);
  },

  async login(credentials: { email: string; password: string }) {
    const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(credentials)
    });
    return handleResponse(response);
  },

  async googleLogin(googleAccessToken: string) {
    const response = await fetch(`${API_BASE_URL}/api/auth/google`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ accessToken: googleAccessToken })
    });
    return handleResponse(response);
  },

  async googleSignup(googleAccessToken: string) {
    // Google OAuth typically uses the same endpoint for both login and signup
    // The backend determines if it's a new user based on the email
    const response = await fetch(`${API_BASE_URL}/api/auth/google`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ accessToken: googleAccessToken })
    });
    return handleResponse(response);
  },

  async setAuthToken(token: string | null) {
    if (token) {
      await AsyncStorage.setItem('authToken', token);
    } else {
      await AsyncStorage.removeItem('authToken');
    }
  },

  // User Profile
  async getUserProfile() {
    const response = await fetch(`${API_BASE_URL}/api/user/profile`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  async updateUserProfile(profileData: Record<string, unknown>) {
    const response = await fetch(`${API_BASE_URL}/api/user/profile`, {
      method: 'PUT',
      headers: await getAuthHeaders(),
      body: JSON.stringify(profileData)
    });
    return handleResponse(response);
  },

  async uploadAvatar(imageUri: string, imageType: 'base64' | 'file' = 'base64') {
    if (imageType === 'base64') {
      // Send base64 directly in JSON
      const response = await fetch(`${API_BASE_URL}/api/user/upload-avatar`, {
        method: 'POST',
        headers: await getAuthHeaders(),
        body: JSON.stringify({ imageData: imageUri, encoding: 'base64' })
      });
      return handleResponse(response);
    } else {
      // Send as multipart/form-data
      const formData = new FormData();
      formData.append('avatar', {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'avatar.jpg',
      } as unknown as Blob);
      
      const token = await AsyncStorage.getItem('authToken');
      const response = await fetch(`${API_BASE_URL}/api/user/upload-avatar`, {
        method: 'POST',
        headers: {
          ...(token && { 'Authorization': `Bearer ${token}` }),
        },
        body: formData
      });
      return handleResponse(response);
    }
  },

  // XP and Coins
  async awardXP(amount: number, reason: string) {
    const response = await fetch(`${API_BASE_URL}/api/user/xp`, {
      method: 'POST',
      headers: await getAuthHeaders(),
      body: JSON.stringify({ xpAmount: amount, reason })
    });
    return handleResponse(response);
  },

  async awardCoins(amount: number, reason: string) {
    const response = await fetch(`${API_BASE_URL}/api/user/coins`, {
      method: 'POST',
      headers: await getAuthHeaders(),
      body: JSON.stringify({ coinAmount: amount, reason })
    });
    return handleResponse(response);
  },

  // Achievements
  async getAchievements() {
    const response = await fetch(`${API_BASE_URL}/api/user/achievements`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  async unlockAchievement(achievementId: number) {
    const response = await fetch(`${API_BASE_URL}/api/user/achievements/unlock`, {
      method: 'POST',
      headers: await getAuthHeaders(),
      body: JSON.stringify({ achievementId })
    });
    return handleResponse(response);
  },

  // Quests
  async getQuests() {
    const response = await fetch(`${API_BASE_URL}/api/user/quests`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  async completeQuest(questId: number) {
    const response = await fetch(`${API_BASE_URL}/api/user/quest/${questId}/complete`, {
      method: 'POST',
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Leaderboard
  async getGlobalLeaderboard(filters: Record<string, string> = {}) {
    const params = new URLSearchParams(filters);
    const response = await fetch(`${API_BASE_URL}/api/leaderboard/global?${params}`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  async getFriendsLeaderboard() {
    const response = await fetch(`${API_BASE_URL}/api/leaderboard/friends`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  async getUserRank(userId: number) {
    const response = await fetch(`${API_BASE_URL}/api/leaderboard/user/${userId}`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Game Sessions
  async createGameSession(params: { gameType: string; symbols?: string[] | string; difficulty?: string; startingCash?: number }) {
    const response = await fetch(`${API_BASE_URL}/api/game/create-session`, {
      method: 'POST',
      headers: await getAuthHeaders(),
      body: JSON.stringify(params)
    });
    return handleResponse(response);
  },

  async saveGameSession(sessionId: string, state: Record<string, unknown>) {
    const response = await fetch(`${API_BASE_URL}/api/game/save-session`, {
      method: 'POST',
      headers: await getAuthHeaders(),
      body: JSON.stringify({ sessionId, state })
    });
    return handleResponse(response);
  },

  async resumeGameSession(sessionId: string) {
    const response = await fetch(`${API_BASE_URL}/api/game/resume-session/${sessionId}`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  async completeGameSession(sessionId: string, results: Record<string, unknown>) {
    const response = await fetch(`${API_BASE_URL}/api/game/complete-session`, {
      method: 'POST',
      headers: await getAuthHeaders(),
      body: JSON.stringify({ sessionId, results })
    });
    return handleResponse(response);
  },

  async discardGameSession(sessionId: string) {
    const response = await fetch(`${API_BASE_URL}/api/game/discard-session`, {
      method: 'DELETE',
      headers: await getAuthHeaders(),
      body: JSON.stringify({ sessionId })
    });
    return handleResponse(response);
  },

  // Learning System (DEPRECATED - Use getKnowledgeTopics, getKnowledgeTopic, completeUserLesson, getUserLearningProgress instead)
  /** @deprecated Use getKnowledgeTopics instead */
  async getTopics() {
    // eslint-disable-next-line no-console
    console.warn('getTopics is deprecated. Use getKnowledgeTopics instead.');
    // Fallback to new endpoint
    return this.getKnowledgeTopics();
  },

  /** @deprecated Use getKnowledgeTopic instead */
  async getTopic(topicId: string) {
    // eslint-disable-next-line no-console
    console.warn('getTopic is deprecated. Use getKnowledgeTopic instead.');
    // Fallback to new endpoint
    return this.getKnowledgeTopic(topicId);
  },

  /** @deprecated Use completeUserLesson instead */
  async completeLesson(lessonId: string) {
    // eslint-disable-next-line no-console
    console.warn('completeLesson is deprecated. Use completeUserLesson instead.');
    return { success: false, message: 'This endpoint is deprecated. Use completeUserLesson instead.' };
  },

  /** @deprecated This endpoint may not be implemented */
  async completeTopic(topicId: string) {
    // eslint-disable-next-line no-console
    console.warn('completeTopic is deprecated and may not be implemented.');
    return { success: false, message: 'This endpoint is deprecated.' };
  },

  /** @deprecated Use getUserLearningProgress instead */
  async getLearningProgress() {
    // eslint-disable-next-line no-console
    console.warn('getLearningProgress is deprecated. Use getUserLearningProgress instead.');
    // Fallback to new endpoint
    return this.getUserLearningProgress();
  },

  // Knowledge & Learning Functions (Chatbot Integration)
  async getKnowledgeTopics(category?: string, difficulty?: string) {
    const params = new URLSearchParams();
    if (category) params.append('category', category);
    if (difficulty) params.append('difficulty', difficulty);
    
    const response = await fetch(`${CHATBOT_URL}/context/knowledge/topics?${params.toString()}`, {
      headers: { 'Content-Type': 'application/json' },
    });
    return handleResponse(response);
  },

  async getKnowledgeTopic(topicId: string) {
    const response = await fetch(`${CHATBOT_URL}/context/knowledge/topics/${topicId}`, {
      headers: { 'Content-Type': 'application/json' },
    });
    return handleResponse(response);
  },

  async completeUserLesson(lessonData: { lessonId: string, topicId: string, timeSpent?: number, score?: number }) {
    const response = await fetch(`${API_BASE_URL}/api/user/complete-lesson`, {
      method: 'POST',
      headers: await getAuthHeaders(),
      body: JSON.stringify(lessonData)
    });
    return handleResponse(response);
  },

  async getUserLearningProgress() {
    const response = await fetch(`${API_BASE_URL}/api/user/learning-progress`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Portfolio
  async getPortfolio() {
    const response = await fetch(`${API_BASE_URL}/api/portfolio`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  async getPortfolioItems() {
    const response = await fetch(`${API_BASE_URL}/api/portfolio/items`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  async getTrades() {
    const response = await fetch(`${API_BASE_URL}/api/portfolio/trades`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  async getPositions() {
    const response = await fetch(`${API_BASE_URL}/api/portfolio/positions`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Portfolio Management
  async createPortfolio(portfolioData: Record<string, unknown>) {
    const response = await fetch(`${API_BASE_URL}/api/portfolio`, {
      method: 'POST',
      headers: await getAuthHeaders(),
      body: JSON.stringify(portfolioData)
    });
    return handleResponse(response);
  },

  async updatePortfolio(portfolioId: string, portfolioData: Record<string, unknown>) {
    const response = await fetch(`${API_BASE_URL}/api/portfolio/${portfolioId}`, {
      method: 'PUT',
      headers: await getAuthHeaders(),
      body: JSON.stringify(portfolioData)
    });
    return handleResponse(response);
  },

  async deletePortfolio(portfolioId: string) {
    const response = await fetch(`${API_BASE_URL}/api/portfolio/${portfolioId}`, {
      method: 'DELETE',
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  async createPortfolioFromSignal(signalId: number, portfolioName: string, investmentAmount: number) {
    const response = await fetch(`${API_BASE_URL}/api/portfolio/from-signal`, {
      method: 'POST',
      headers: await getAuthHeaders(),
      body: JSON.stringify({ signalId, portfolioName, investmentAmount })
    });
    return handleResponse(response);
  },

  async getPortfolioPerformance(portfolioId: string) {
    const response = await fetch(`${API_BASE_URL}/api/portfolio/${portfolioId}/performance`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Trading Signals
  async getTopSignals(assetClass: string | null = null, limit: number = 3) {
    const params = new URLSearchParams();
    if (assetClass) params.append('assetType', assetClass);
    params.append('limit', limit.toString());
    
    const response = await fetch(`${API_BASE_URL}/api/signals/top?${params}`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  async getHistoricalSignals(filters: { limit?: number; assetType?: string; outcome?: string; offset?: number } = {}) {
    const params = new URLSearchParams();
    if (filters.limit) params.append('limit', filters.limit.toString());
    if (filters.assetType) params.append('assetType', filters.assetType);
    if (filters.outcome) params.append('outcome', filters.outcome);
    if (filters.offset) params.append('offset', filters.offset.toString());
    
    const response = await fetch(`${API_BASE_URL}/api/signals/historical?${params}`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Game Sessions
  async getActiveSessions() {
    const response = await fetch(`${API_BASE_URL}/api/game/sessions`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  async getGameHistory(limit: number = 10) {
    const params = new URLSearchParams();
    params.append('limit', limit.toString());
    
    const response = await fetch(`${API_BASE_URL}/api/game/history?${params}`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Market Data
  async getMarketData(symbol: string) {
    const response = await fetch(`${API_BASE_URL}/api/market-data/real-time/${symbol}`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Chatbot
  async sendChatMessage(message: string, userId: string = 'anonymous') {
    const response = await fetch(`${CHATBOT_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, userId })
    });
    return handleResponse(response);
  },

  // Onboarding
  async startOnboarding(userData: {
    firstName: string;
    email: string;
    userId: number;
  }) {
    const response = await fetch(`${API_BASE_URL}/api/chatbot/onboarding/start`, {
      method: 'POST',
      headers: await getAuthHeaders(),
      body: JSON.stringify(userData)
    });
    return handleResponse(response);
  },

  async sendOnboardingResponse(data: {
    sessionId: string;
    answer: string;
    conversationHistory?: unknown[];
    userAnswers?: unknown[];
  }) {
    const response = await fetch(`${API_BASE_URL}/api/chatbot/onboarding/respond`, {
      method: 'POST',
      headers: await getAuthHeaders(),
      body: JSON.stringify(data)
    });
    return handleResponse(response);
  },

  async completeOnboarding(data: {
    sessionId: string;
    conversationHistory?: unknown[];
    userAnswers?: unknown[];
    userProfile?: Record<string, unknown>;
  }) {
    const response = await fetch(`${API_BASE_URL}/api/user/complete-onboarding`, {
      method: 'POST',
      headers: await getAuthHeaders(),
      body: JSON.stringify(data)
    });
    return handleResponse(response);
  },

  // Notifications Functions
  async getNotifications(filters?: { status?: string, type?: string, limit?: number, offset?: number }) {
    const params = new URLSearchParams();
    if (filters?.status) params.append('status', filters.status);
    if (filters?.type) params.append('type', filters.type);
    if (filters?.limit) params.append('limit', filters.limit.toString());
    if (filters?.offset) params.append('offset', filters.offset.toString());
    
    const response = await fetch(`${API_BASE_URL}/api/notifications?${params.toString()}`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  async getUnreadNotificationCount() {
    const response = await fetch(`${API_BASE_URL}/api/notifications/unread-count`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  async markNotificationRead(notificationId: string) {
    const response = await fetch(`${API_BASE_URL}/api/notifications/${notificationId}/read`, {
      method: 'PUT',
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  async markAllNotificationsRead() {
    const response = await fetch(`${API_BASE_URL}/api/notifications/read-all`, {
      method: 'PUT',
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  async deleteNotification(notificationId: string) {
    const response = await fetch(`${API_BASE_URL}/api/notifications/${notificationId}`, {
      method: 'DELETE',
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Analytics Functions
  async getAnalyticsPerformance(timeframe?: string, portfolioId?: string) {
    const params = new URLSearchParams();
    if (timeframe) params.append('timeframe', timeframe);
    if (portfolioId) params.append('portfolioId', portfolioId);
    
    const response = await fetch(`${API_BASE_URL}/api/analytics/performance?${params.toString()}`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // News Functions
  async getTrendingMarketData(limit?: number, timeframe?: string) {
    const params = new URLSearchParams();
    if (limit) params.append('limit', limit.toString());
    if (timeframe) params.append('timeframe', timeframe);
    
    const response = await fetch(`${API_BASE_URL}/api/market-data/trending?${params.toString()}`, {
      headers: await getAuthHeaders(),
    });
    return handleResponse(response);
  },

  async searchNews(query: string, limit?: number) {
    const params = new URLSearchParams();
    params.append('q', query);
    if (limit) params.append('k', limit.toString());
    
    const response = await fetch(`${CHATBOT_URL}/v1/search?${params.toString()}`, {
      headers: { 'Content-Type': 'application/json' },
    });
    return handleResponse(response);
  },

  // Onboarding Analytics
  async trackOnboardingAnalytics(analyticsData: Record<string, unknown>) {
    const response = await fetch(`${API_BASE_URL}/api/analytics/onboarding`, {
      method: 'POST',
      headers: await getAuthHeaders(),
      body: JSON.stringify(analyticsData)
    });
    return handleResponse(response);
  },

  // Complete Onboarding (Alias for existing)
  async completeUserOnboarding(data: {
    sessionId: string;
    conversationHistory?: unknown[];
    userAnswers?: unknown[];
    userProfile?: Record<string, unknown>;
  }) {
    return this.completeOnboarding(data);
  },
};

// Export individual functions for easier imports
export const {
  signup,
  login,
  googleLogin,
  googleSignup,
  setAuthToken,
  getUserProfile,
  updateUserProfile,
  uploadAvatar,
  awardXP,
  awardCoins,
  getAchievements,
  unlockAchievement,
  getQuests,
  completeQuest,
  getGlobalLeaderboard,
  getFriendsLeaderboard,
  getUserRank,
  createGameSession,
  saveGameSession,
  resumeGameSession,
  completeGameSession,
  discardGameSession,
  getTopics,
  getTopic,
  completeLesson,
  completeTopic,
  getLearningProgress,
  getPortfolio,
  getPortfolioItems,
  getTrades,
  getPositions,
  createPortfolio,
  updatePortfolio,
  deletePortfolio,
  createPortfolioFromSignal,
  getPortfolioPerformance,
  getTopSignals,
  getHistoricalSignals,
  getActiveSessions,
  getGameHistory,
  getMarketData,
  sendChatMessage,
  startOnboarding,
  sendOnboardingResponse,
  completeOnboarding,
  getKnowledgeTopics,
  getKnowledgeTopic,
  completeUserLesson,
  getUserLearningProgress,
  getNotifications,
  getUnreadNotificationCount,
  markNotificationRead,
  markAllNotificationsRead,
  deleteNotification,
  getAnalyticsPerformance,
  getTrendingMarketData,
  searchNews,
  completeUserOnboarding,
} = apiService;

export default apiService;
