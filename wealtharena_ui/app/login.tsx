import React, { useState } from 'react';
import { View, StyleSheet, ScrollView, KeyboardAvoidingView, Platform, Pressable, Image, Alert } from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTheme, Text, Button, TextInput, Card, Icon, FoxMascot, tokens } from '@/src/design-system';
import { login } from '@/services/apiService';
import { useUser } from '@/contexts/UserContext';

export default function LoginScreen() {
  const router = useRouter();
  const { theme } = useTheme();
  const { login: loginUser } = useUser();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [emailError, setEmailError] = useState('');
  const [passwordError, setPasswordError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const validateEmail = (email: string) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const handleLogin = async () => {
    // Reset errors
    setEmailError('');
    setPasswordError('');

    // Validate
    let hasError = false;
    if (!email) {
      setEmailError('Email is required');
      hasError = true;
    } else if (!validateEmail(email)) {
      setEmailError('Please enter a valid email');
      hasError = true;
    }
    
    if (!password) {
      setPasswordError('Password is required');
      hasError = true;
    } else if (password.length < 6) {
      setPasswordError('Password must be at least 6 characters');
      hasError = true;
    }

    if (hasError) return;

    // Call backend API
    setIsLoading(true);
    try {
      const response = await login({ email, password });
      
      if (response.success && response.data) {
        // Save user data to context
        await loginUser(response.data.user, response.data.token);
        
        Alert.alert('Success', `Welcome back, ${response.data.user.username}!`);
        console.log('Login successful:', response.data);
        router.replace('/(tabs)/dashboard');
      } else {
        Alert.alert('Login Failed', response.message);
      }
    } catch (error: any) {
      console.error('Login error:', error);
      Alert.alert('Error', error.message || 'Failed to login. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleSignIn = () => {
    console.log('Google Sign In');
    // Google sign in - existing users go to dashboard
    router.replace('/(tabs)/dashboard');
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.bg }]} edges={['top']}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {/* Hero Section with Mascot */}
          <View style={styles.hero}>
            <FoxMascot variant="confident" size={100} />
            <Text variant="h1" weight="bold" center>
              Welcome Back!
            </Text>
            <Text variant="body" center muted style={styles.subtitle}>
              Sign in to continue your trading journey
            </Text>
          </View>

          {/* Login Form Card */}
          <Card style={styles.formCard} elevation="med">
            {/* Social Login */}
            <Button
              variant="secondary"
              size="large"
              onPress={handleGoogleSignIn}
              fullWidth
              icon={<Image source={{ uri: 'https://www.google.com/images/branding/googleg/1x/googleg_standard_color_128dp.png' }} style={{ width: 20, height: 20 }} />}
            >
              Continue with Google
            </Button>

            {/* Divider */}
            <View style={styles.divider}>
              <View style={[styles.dividerLine, { backgroundColor: theme.border }]} />
              <Text variant="small" muted style={styles.dividerText}>OR</Text>
              <View style={[styles.dividerLine, { backgroundColor: theme.border }]} />
            </View>

            {/* Input Fields */}
            <TextInput
              label="Email"
              placeholder="your.email@example.com"
              value={email}
              onChangeText={(text) => {
                setEmail(text);
                if (emailError) setEmailError('');
              }}
              keyboardType="email-address"
              autoCapitalize="none"
              autoComplete="email"
              error={emailError}
            />

            <TextInput
              label="Password"
              placeholder="Enter your password"
              value={password}
              onChangeText={(text) => {
                setPassword(text);
                if (passwordError) setPasswordError('');
              }}
              secureTextEntry={!showPassword}
              autoCapitalize="none"
              rightIcon={<Text variant="small" color={theme.primary}>
                {showPassword ? 'Hide' : 'Show'}
              </Text>}
              onRightIconPress={() => setShowPassword(!showPassword)}
              error={passwordError}
            />

            {/* Forgot Password */}
            <Pressable style={styles.forgotPassword}>
              <Text variant="small" color={theme.primary} weight="semibold">
                Forgot Password?
              </Text>
            </Pressable>

            {/* Login Button */}
            <Button
              variant="primary"
              size="large"
              onPress={handleLogin}
              fullWidth
              loading={isLoading}
              disabled={isLoading}
              icon={!isLoading ? <Icon name="check-shield" size={20} color={theme.bg} /> : undefined}
            >
              {isLoading ? 'Signing In...' : 'Sign In'}
            </Button>
          </Card>

          {/* Footer */}
          <View style={styles.footer}>
            <Text variant="small" muted>Don't have an account? </Text>
            <Pressable onPress={() => router.push('/signup')}>
              <Text variant="small" weight="bold" color={theme.primary}>
                Sign Up
              </Text>
            </Pressable>
          </View>

          {/* Security Note */}
          <View style={styles.securityNote}>
            <Icon name="shield" size={16} color={theme.muted} />
            <Text variant="xs" muted style={styles.securityText}>
              Your data is encrypted and secure
            </Text>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  keyboardView: {
    flex: 1,
  },
  scrollContent: {
    padding: tokens.spacing.lg,
    paddingTop: tokens.spacing.md,
  },
  hero: {
    alignItems: 'center',
    marginBottom: tokens.spacing.xl,
  },
  subtitle: {
    marginTop: tokens.spacing.sm,
  },
  formCard: {
    gap: tokens.spacing.md,
    marginBottom: tokens.spacing.md,
  },
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: tokens.spacing.sm,
  },
  dividerLine: {
    flex: 1,
    height: 1,
  },
  dividerText: {
    marginHorizontal: tokens.spacing.md,
  },
  forgotPassword: {
    alignSelf: 'flex-end',
    marginTop: -tokens.spacing.xs,
  },
  footer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: tokens.spacing.md,
  },
  securityNote: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: tokens.spacing.xs,
    marginTop: tokens.spacing.lg,
    marginBottom: tokens.spacing.xl,
  },
  securityText: {
    opacity: 0.7,
  },
});
