import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Image, Platform } from 'react-native';
import { useRouter, Link } from 'expo-router';
import Colors from '@/constants/colors';
import { LinearGradient } from 'expo-linear-gradient';
import { Menu, ArrowRight, Rocket, Shield, Stars, Sparkles, LogIn, UserPlus } from 'lucide-react-native';

export default function Landing() {
  const router = useRouter();

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <View style={styles.topBar}>
        <Image source={{ uri: 'https://images.unsplash.com/photo-1568605114967-8130f3a36994?q=80&w=200&auto=format&fit=crop' }} style={styles.avatar} />
        <View style={styles.topCenter}>
          <Text style={styles.topTitle}>WealthArena</Text>
          <Text style={styles.topSubtitle}>Playful investing, serious outcomes</Text>
        </View>
        <View style={styles.menuWrap}>
          <Menu color={Colors.text} size={22} />
          <View style={styles.notifBubble}><Text style={styles.notifText}>2</Text></View>
        </View>
      </View>

      <LinearGradient colors={[Colors.surface, Colors.surfaceLight]} style={styles.hero}>
        <View style={styles.heroBadge}><Sparkles size={14} color={Colors.text} /></View>
        <Text style={styles.heroTitle}>Level up your investing</Text>
        <Text style={styles.heroSub}>Avatars, levels, and rewards blended with a pro-grade portfolio hub.</Text>
        <View style={styles.heroCtas}>
          <Link href="/signup" asChild>
            <TouchableOpacity style={styles.primaryBtn} testID="landing-signup">
              <UserPlus size={16} color={Colors.primary} />
              <Text style={styles.primaryBtnText}>Create account</Text>
            </TouchableOpacity>
          </Link>
          <Link href="/login" asChild>
            <TouchableOpacity style={styles.secondaryBtn} testID="landing-login">
              <LogIn size={16} color={Colors.text} />
              <Text style={styles.secondaryBtnText}>Sign in</Text>
            </TouchableOpacity>
          </Link>
        </View>
      </LinearGradient>

      <View style={styles.cards}>
        <LinearGradient colors={[Colors.glow.blue, 'transparent']} style={styles.card}>
          <View style={[styles.cardIcon, { backgroundColor: Colors.glow.blue }]}><Rocket size={18} color={Colors.secondary} /></View>
          <Text style={styles.cardTitle}>Guided onboarding</Text>
          <Text style={styles.cardText}>Chatbot asks a few smart questions to tailor your path.</Text>
          <TouchableOpacity style={styles.cardLink} onPress={() => router.push('/onboarding')}>
            <Text style={styles.cardLinkText}>Try the wizard</Text>
            <ArrowRight size={16} color={Colors.text} />
          </TouchableOpacity>
        </LinearGradient>

        <LinearGradient colors={[Colors.glow.purple, 'transparent']} style={styles.card}>
          <View style={[styles.cardIcon, { backgroundColor: Colors.glow.purple }]}><Stars size={18} color={Colors.accent} /></View>
          <Text style={styles.cardTitle}>Gamified dashboard</Text>
          <Text style={styles.cardText}>Levels, XP, collectibles and glowing cards on dark UI.</Text>
          <TouchableOpacity style={styles.cardLink} onPress={() => router.push('/(tabs)/dashboard')}>
            <Text style={styles.cardLinkText}>Open dashboard</Text>
            <ArrowRight size={16} color={Colors.text} />
          </TouchableOpacity>
        </LinearGradient>

        <LinearGradient colors={[Colors.glow.gold || 'rgba(255,204,0,0.9)', 'transparent']} style={styles.card}>
          <View style={[styles.cardIcon, { backgroundColor: Colors.glow.gold }]}><Shield size={18} color={Colors.gold} /></View>
          <Text style={styles.cardTitle}>No integrations yet</Text>
          <Text style={styles.cardText}>This build is offline-first for preview. We can wire APIs later.</Text>
          <View style={styles.cardGhost} />
        </LinearGradient>
      </View>

      <View style={{ height: 40 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  content: { padding: 20, gap: 16 },
  topBar: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  avatar: { width: 40, height: 40, borderRadius: 20 },
  topCenter: { flex: 1 },
  topTitle: { color: Colors.text, fontSize: 16, fontWeight: '700' as const },
  topSubtitle: { color: Colors.textSecondary, fontSize: 12 },
  menuWrap: { width: 40, height: 40, borderRadius: 20, backgroundColor: Colors.surface, alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: Colors.border, position: 'relative' },
  notifBubble: { position: 'absolute', top: -2, right: -2, width: 18, height: 18, borderRadius: 9, backgroundColor: '#FF3B9A', alignItems: 'center', justifyContent: 'center', shadowColor: '#FF3B9A', shadowOpacity: Platform.OS === 'web' ? 1 : 0.9, shadowRadius: 6 },
  notifText: { color: Colors.text, fontSize: 10, fontWeight: '700' as const },

  hero: { borderRadius: 24, padding: 20, gap: 12, borderWidth: 1, borderColor: Colors.border },
  heroBadge: { alignSelf: 'flex-start', backgroundColor: Colors.primaryLight, paddingHorizontal: 10, paddingVertical: 6, borderRadius: 16 },
  heroTitle: { color: Colors.text, fontSize: 22, fontWeight: '700' as const },
  heroSub: { color: Colors.textSecondary, fontSize: 14 },
  heroCtas: { flexDirection: 'row', gap: 12, paddingTop: 6 },
  primaryBtn: { flexDirection: 'row', gap: 8, backgroundColor: Colors.gold, paddingHorizontal: 16, paddingVertical: 12, borderRadius: 16 },
  primaryBtnText: { color: Colors.primary, fontWeight: '700' as const, fontSize: 14 },
  secondaryBtn: { flexDirection: 'row', gap: 8, backgroundColor: Colors.surface, borderWidth: 1, borderColor: Colors.border, paddingHorizontal: 16, paddingVertical: 12, borderRadius: 16 },
  secondaryBtnText: { color: Colors.text, fontWeight: '600' as const, fontSize: 14 },

  cards: { gap: 12 },
  card: { borderRadius: 20, padding: 16, gap: 8, borderWidth: 1, borderColor: Colors.border },
  cardIcon: { width: 36, height: 36, borderRadius: 18, alignItems: 'center', justifyContent: 'center' },
  cardTitle: { color: Colors.text, fontSize: 16, fontWeight: '700' as const },
  cardText: { color: Colors.textSecondary, fontSize: 12 },
  cardLink: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 6 },
  cardLinkText: { color: Colors.text, fontWeight: '600' as const },
  cardGhost: { height: 0 },
});
