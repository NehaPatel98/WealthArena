/**
 * API Service - Frontend to Backend Integration
 * Handles all HTTP requests to the WealthArena Backend
 */

// Use your computer's local IP address instead of localhost for Expo to connect
// Update this IP if you change networks
const API_BASE_URL = __DEV__ 
  ? 'https://wealtharena-backend-7189.azurewebsites.net/api' 
  : 'https://wealtharena-backend-7189.azurewebsites.net/api';

// Store authentication token
let authToken: string | null = null;

/**
 * Set the authentication token
 */
export function setAuthToken(token: string | null) {
  authToken = token;
}

/**
 * Get the current authentication token
 */
export function getAuthToken() {
  return authToken;
}

/**
 * Make an API request
 */
async function apiRequest<T = any>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  // Add authorization header if token exists
  if (authToken) {
    (headers as any)['Authorization'] = `Bearer ${authToken}`;
  }

  const url = `${API_BASE_URL}${endpoint}`;

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000000); // 30000 second timeout
    
    const response = await fetch(url, {
      ...options,
      headers,
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || `API Error: ${response.status}`);
    }

    return data;
  } catch (error) {
    console.error('API Request Error:', error);
    throw error;
  }
}

// ============================================
// Authentication API
// ============================================

export interface SignupData {
  username: string;
  email: string;
  password: string;
  full_name?: string;
}

export interface LoginData {
  email: string;
  password: string;
}

export interface AuthResponse {
  success: boolean;
  message: string;
  data?: {
    token: string;
    user: {
      user_id: number;
      username: string;
      email: string;
      full_name: string;
      tier_level: string;
      xp_points: number;
      total_balance: number;
    };
  };
}

export async function signup(data: SignupData): Promise<AuthResponse> {
  const response = await apiRequest<AuthResponse>('/auth/signup', {
    method: 'POST',
    body: JSON.stringify(data),
  });

  // Store token if successful
  if (response.success && response.data?.token) {
    setAuthToken(response.data.token);
  }

  return response;
}

export async function login(data: LoginData): Promise<AuthResponse> {
  const response = await apiRequest<AuthResponse>('/auth/login', {
    method: 'POST',
    body: JSON.stringify(data),
  });

  // Store token if successful
  if (response.success && response.data?.token) {
    setAuthToken(response.data.token);
  }

  return response;
}

export async function logout() {
  setAuthToken(null);
}

// ============================================
// User API
// ============================================

export interface UserProfile {
  user_id: number;
  username: string;
  email: string;
  full_name: string;
  tier_level: string;
  xp_points: number;
  total_balance: number;
  avatar_url?: string;
  bio?: string;
  created_at: string;
}

export async function getUserProfile(): Promise<UserProfile> {
  const response = await apiRequest<{ success: boolean; data: UserProfile }>('/user/profile');
  return response.data;
}

export interface UpdateProfileData {
  firstName?: string;
  lastName?: string;
  username?: string;
  displayName?: string;
  bio?: string;
  avatarUrl?: string;
  avatar_type?: 'mascot' | 'custom';
  avatar_variant?: string;
}

export async function updateUserProfile(data: UpdateProfileData): Promise<UserProfile> {
  const response = await apiRequest<{ success: boolean; data: UserProfile }>('/user/profile', {
    method: 'PUT',
    body: JSON.stringify(data),
  });
  return response.data;
}

export async function updateUserXP(xpChange: number, reason: string) {
  return apiRequest('/user/xp', {
    method: 'POST',
    body: JSON.stringify({ xpChange, reason }),
  });
}

export async function getUserAchievements() {
  const response = await apiRequest<{ success: boolean; data: any[] }>('/user/achievements');
  return response.data;
}

export async function getUserQuests() {
  const response = await apiRequest<{ success: boolean; data: any[] }>('/user/quests');
  return response.data;
}

export async function getLeaderboard(limit: number = 100) {
  const response = await apiRequest<{ success: boolean; data: any[] }>(`/user/leaderboard?limit=${limit}`);
  return response.data;
}

// ============================================
// AI Signals API
// ============================================

export interface AISignal {
  signal_id: number;
  symbol: string;
  signal_type: string;
  confidence_score: number;
  entry_price: number;
  target_price: number;
  stop_loss: number;
  timeframe: string;
  rationale: string;
  created_at: string;
  expires_at: string;
  is_active: boolean;
}

export async function getTopSignals(limit: number = 10): Promise<AISignal[]> {
  const response = await apiRequest<{ success: boolean; data: AISignal[] }>(`/signals/top?limit=${limit}`);
  return response.data;
}

export async function getSignalById(signalId: number): Promise<AISignal> {
  const response = await apiRequest<{ success: boolean; data: AISignal }>(`/signals/${signalId}`);
  return response.data;
}

export async function getSignalsBySymbol(symbol: string): Promise<AISignal[]> {
  const response = await apiRequest<{ success: boolean; data: AISignal[] }>(`/signals/symbol/${symbol}`);
  return response.data;
}

// ============================================
// Portfolio API
// ============================================

export interface PortfolioOverview {
  total_balance: number;
  total_pnl: number;
  total_pnl_percentage: number;
  active_positions: number;
  total_trades: number;
  win_rate: number;
}

export interface PortfolioItem {
  item_id: number;
  symbol: string;
  item_type: string;
  quantity: number;
  average_price: number;
  current_price: number;
  total_value: number;
  unrealized_pnl: number;
  unrealized_pnl_percentage: number;
}

export interface Trade {
  trade_id: number;
  symbol: string;
  trade_type: string;
  quantity: number;
  entry_price: number;
  exit_price?: number;
  pnl?: number;
  status: string;
  executed_at: string;
}

export async function getPortfolioOverview(): Promise<PortfolioOverview> {
  const response = await apiRequest<{ success: boolean; data: PortfolioOverview }>('/portfolio');
  return response.data;
}

export async function getPortfolioItems(): Promise<PortfolioItem[]> {
  const response = await apiRequest<{ success: boolean; data: PortfolioItem[] }>('/portfolio/items');
  return response.data;
}

export async function getUserTrades(): Promise<Trade[]> {
  const response = await apiRequest<{ success: boolean; data: Trade[] }>('/portfolio/trades');
  return response.data;
}

export async function getActivePositions(): Promise<PortfolioItem[]> {
  const response = await apiRequest<{ success: boolean; data: PortfolioItem[] }>('/portfolio/positions');
  return response.data;
}

// ============================================
// Chat API
// ============================================

export interface ChatSession {
  session_id: number;
  session_name: string;
  created_at: string;
  message_count: number;
}

export interface ChatMessage {
  message_id: number;
  session_id: number;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export async function createChatSession(sessionName?: string): Promise<ChatSession> {
  const response = await apiRequest<{ success: boolean; data: ChatSession }>('/chat/session', {
    method: 'POST',
    body: JSON.stringify({ session_name: sessionName }),
  });
  return response.data;
}

export async function sendChatMessage(sessionId: number, message: string): Promise<ChatMessage> {
  const response = await apiRequest<{ success: boolean; data: ChatMessage }>('/chat/message', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  return response.data;
}

export async function getChatHistory(sessionId: number): Promise<ChatMessage[]> {
  const response = await apiRequest<{ success: boolean; data: ChatMessage[] }>(`/chat/history?session_id=${sessionId}`);
  return response.data;
}

export async function getChatSessions(): Promise<ChatSession[]> {
  const response = await apiRequest<{ success: boolean; data: ChatSession[] }>('/chat/sessions');
  return response.data;
}

export async function submitChatFeedback(messageId: number, rating: number, comment?: string) {
  return apiRequest('/chat/feedback', {
    method: 'POST',
    body: JSON.stringify({ message_id: messageId, rating, comment }),
  });
}

// ============================================
// Health Check
// ============================================

export async function checkHealth() {
  try {
    const response = await apiRequest('/health');
    return response;
  } catch (error) {
    console.error('Health check failed:', error);
    return { success: false, message: 'Backend is not reachable' };
  }
}

// ============================================
// Generic API helpers (for services expecting axios-like apiService)
// ============================================

export const apiService = {
  get: async <T = any>(endpoint: string, options: RequestInit = {}) => {
    const data = await apiRequest<T>(endpoint, { ...options, method: 'GET' });
    return { data } as { data: T };
  },
  post: async <T = any>(endpoint: string, body?: any, options: RequestInit = {}) => {
    const data = await apiRequest<T>(endpoint, {
      ...options,
      method: 'POST',
      body: body !== undefined ? JSON.stringify(body) : options.body,
    });
    return { data } as { data: T };
  },
  put: async <T = any>(endpoint: string, body?: any, options: RequestInit = {}) => {
    const data = await apiRequest<T>(endpoint, {
      ...options,
      method: 'PUT',
      body: body !== undefined ? JSON.stringify(body) : options.body,
    });
    return { data } as { data: T };
  },
  delete: async <T = any>(endpoint: string, options: RequestInit = {}) => {
    const data = await apiRequest<T>(endpoint, { ...options, method: 'DELETE' });
    return { data } as { data: T };
  },
};

export default {
  // Auth
  signup,
  login,
  logout,
  setAuthToken,
  getAuthToken,
  
  // User
  getUserProfile,
  updateUserProfile,
  updateUserXP,
  getUserAchievements,
  getUserQuests,
  getLeaderboard,
  
  // Signals
  getTopSignals,
  getSignalById,
  getSignalsBySymbol,
  
  // Portfolio
  getPortfolioOverview,
  getPortfolioItems,
  getUserTrades,
  getActivePositions,
  
  // Chat
  createChatSession,
  sendChatMessage,
  getChatHistory,
  getChatSessions,
  submitChatFeedback,
  
  // Health
  checkHealth,
};

