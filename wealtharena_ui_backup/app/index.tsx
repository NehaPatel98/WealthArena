import React from 'react';
import { Redirect } from 'expo-router';
import { useUserTier } from '@/contexts/UserTierContext';

export default function Index() {
  const { profile, isLoading } = useUserTier();

  if (isLoading) return null;

  const isAuthenticated = false;

  if (!isAuthenticated) {
    return <Redirect href="/splash" />;
  }

  if (!profile.tier) {
    return <Redirect href="/onboarding" />;
  }

  return <Redirect href="/(tabs)/dashboard" />;
}
