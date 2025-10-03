import React, { useEffect } from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity, Platform } from 'react-native';
import { useRouter } from 'expo-router';
import Colors from '@/constants/colors';
import { LinearGradient } from 'expo-linear-gradient';

export default function SplashScreenWealth() {
  const router = useRouter();

  useEffect(() => {
    const t = setTimeout(() => {
      router.replace('/landing' as any);
    }, 1400);
    return () => clearTimeout(t);
  }, [router]);

  return (
    <View style={styles.container} testID="splash-root">
      <LinearGradient colors={[Colors.backgroundGradientStart, Colors.backgroundGradientEnd]} style={styles.gradient}>
        <View style={styles.logoWrap}>
          <LinearGradient colors={[Colors.accent, Colors.secondary]} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} style={styles.logoOrb}>
            <Text style={styles.logoGlyph}>WA</Text>
          </LinearGradient>
          <View style={styles.badge}>
            <Text style={styles.badgeText}>2</Text>
          </View>
        </View>
        <Text style={styles.title}>WealthArena</Text>
        <Text style={styles.subtitle}>Playful investing. Powerful results.</Text>
        <TouchableOpacity style={styles.cta} onPress={() => router.replace('/landing')} testID="splash-cta">
          <Text style={styles.ctaText}>Enter</Text>
        </TouchableOpacity>
      </LinearGradient>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  gradient: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 24 },
  logoWrap: { position: 'relative', marginBottom: 20 },
  logoOrb: { width: 120, height: 120, borderRadius: 60, alignItems: 'center', justifyContent: 'center', shadowColor: '#000', shadowOpacity: 0.4, shadowRadius: 16 },
  logoGlyph: { fontSize: 36, fontWeight: '700' as const, color: Colors.text },
  badge: { position: 'absolute', top: 8, right: 8, width: 24, height: 24, borderRadius: 12, backgroundColor: '#FF3B9A', alignItems: 'center', justifyContent: 'center', shadowColor: '#FF3B9A', shadowOpacity: Platform.OS === 'web' ? 1 : 0.9, shadowRadius: 8 },
  badgeText: { color: Colors.text, fontSize: 12, fontWeight: '700' as const },
  title: { fontSize: 28, color: Colors.text, fontWeight: '700' as const, marginBottom: 6 },
  subtitle: { fontSize: 14, color: Colors.textSecondary, marginBottom: 28 },
  cta: { backgroundColor: Colors.surface, paddingHorizontal: 24, paddingVertical: 12, borderRadius: 24, borderWidth: 1, borderColor: Colors.border },
  ctaText: { color: Colors.text, fontWeight: '700' as const },
});
