/**
 * User Context
 * Manages authenticated user state and data
 */

import React, { createContext, useContext, useState, useEffect, useMemo } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { getUserProfile, setAuthToken } from '@/services/apiService';
import { FoxVariant } from '@/src/design-system/mascots';

export interface User {
  user_id: number;
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  full_name: string;
  tier_level: string;
  xp_points: number;
  total_balance: number;
  avatar_url?: string;
  avatar_type?: 'mascot' | 'custom';
  avatar_variant?: FoxVariant;
  displayName?: string;
  bio?: string;
  current_level?: number;
  total_coins?: number;
  win_rate?: number;
  total_trades?: number;
  current_streak?: number;
}

interface UserContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (userData: User, token: string) => Promise<void>;
  logout: () => Promise<void>;
  updateUser: (updates: Partial<User>) => void;
  refreshUser: () => Promise<void>;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

export function UserProvider({ children }: Readonly<{ children: React.ReactNode }>) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Load user from storage on mount
  useEffect(() => {
    loadUserFromStorage();
  }, []);

  const loadUserFromStorage = async () => {
    try {
      const [storedUser, storedToken] = await Promise.all([
        AsyncStorage.getItem('user'),
        AsyncStorage.getItem('authToken'),
      ]);

      if (storedUser && storedToken) {
        const userData = JSON.parse(storedUser);
        setUser(userData);
        setAuthToken(storedToken);
        setIsAuthenticated(true);
      }
    } catch (error) {
      console.error('Failed to load user from storage:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (userData: User, token: string) => {
    try {
      // Parse first and last name from full_name if not already set
      if (!userData.firstName && userData.full_name) {
        const nameParts = userData.full_name.trim().split(' ');
        userData.firstName = nameParts[0] || '';
        userData.lastName = nameParts.slice(1).join(' ') || '';
      }

      // Set default avatar if not already set
      if (!userData.avatar_type) {
        userData.avatar_type = 'mascot';
        userData.avatar_variant = 'neutral';
      }

      await Promise.all([
        AsyncStorage.setItem('user', JSON.stringify(userData)),
        AsyncStorage.setItem('authToken', token),
      ]);

      setUser(userData);
      setAuthToken(token);
      setIsAuthenticated(true);
    } catch (error) {
      console.error('Failed to save user data:', error);
      throw error;
    }
  };

  const logout = async () => {
    try {
      await Promise.all([
        AsyncStorage.removeItem('user'),
        AsyncStorage.removeItem('authToken'),
      ]);

      setUser(null);
      setAuthToken(null);
      setIsAuthenticated(false);
    } catch (error) {
      console.error('Failed to logout:', error);
      throw error;
    }
  };

  const updateUser = (updates: Partial<User>) => {
    if (!user) return;

    const updatedUser = { ...user, ...updates };
    setUser(updatedUser);
    AsyncStorage.setItem('user', JSON.stringify(updatedUser));
  };

  const refreshUser = async () => {
    if (!isAuthenticated) return;

    try {
      const profile = await getUserProfile();
      const updatedUser = {
        ...user!,
        ...profile,
      };
      setUser(updatedUser);
      await AsyncStorage.setItem('user', JSON.stringify(updatedUser));
    } catch (error) {
      console.error('Failed to refresh user data:', error);
    }
  };

  const contextValue = useMemo(
    () => ({
      user,
      isLoading,
      isAuthenticated,
      login,
      logout,
      updateUser,
      refreshUser,
    }),
    [user, isLoading, isAuthenticated]
  );

  return (
    <UserContext.Provider value={contextValue}>
      {children}
    </UserContext.Provider>
  );
}

export function useUser() {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
}

