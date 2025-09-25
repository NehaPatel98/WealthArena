import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { colors } from '../theme/colors';

interface LoginScreenProps {
  onLogin: (email: string, password: string) => Promise<void>;
  onNavigateToSignup: () => void;
}

const LoginScreen: React.FC<LoginScreenProps> = ({ onLogin, onNavigateToSignup }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<{ email?: string; password?: string }>({});

  const isDarkMode = true;
  const c = colors[isDarkMode ? 'dark' : 'light'];

  const validateForm = () => {
    const newErrors: { email?: string; password?: string } = {};

    if (!email.trim()) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(email)) {
      newErrors.email = 'Please enter a valid email';
    }

    if (!password.trim()) {
      newErrors.password = 'Password is required';
    } else if (password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleLogin = async () => {
    if (!validateForm()) return;

    setIsLoading(true);
    try {
      await onLogin(email, password);
    } catch (error) {
      Alert.alert('Login Failed', 'Invalid email or password. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={[styles.container, { backgroundColor: c.background }]}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <ScrollView contentContainerStyle={styles.scrollContainer}>
        {/* Header */}
        <View style={styles.header}>
          <View style={[styles.logoContainer, { backgroundColor: c.primary }]}>
            <Icon name="trending-up" size={32} color={c.background} />
          </View>
          <Text style={[styles.title, { color: c.text }]}>Welcome Back</Text>
          <Text style={[styles.subtitle, { color: c.textMuted }]}>
            Sign in to your WealthArena account
          </Text>
        </View>

        {/* Form */}
        <View style={styles.form}>
          {/* Email Input */}
          <View style={styles.inputContainer}>
            <Text style={[styles.label, { color: c.text }]}>Email</Text>
            <View style={[styles.inputWrapper, { borderColor: errors.email ? c.danger : c.border }]}>
              <Icon name="email-outline" size={20} color={c.textMuted} />
              <TextInput
                style={[styles.input, { color: c.text }]}
                placeholder="Enter your email"
                placeholderTextColor={c.textMuted}
                value={email}
                onChangeText={setEmail}
                keyboardType="email-address"
                autoCapitalize="none"
                autoCorrect={false}
              />
            </View>
            {errors.email && (
              <Text style={[styles.errorText, { color: c.danger }]}>{errors.email}</Text>
            )}
          </View>

          {/* Password Input */}
          <View style={styles.inputContainer}>
            <Text style={[styles.label, { color: c.text }]}>Password</Text>
            <View style={[styles.inputWrapper, { borderColor: errors.password ? c.danger : c.border }]}>
              <Icon name="lock-outline" size={20} color={c.textMuted} />
              <TextInput
                style={[styles.input, { color: c.text }]}
                placeholder="Enter your password"
                placeholderTextColor={c.textMuted}
                value={password}
                onChangeText={setPassword}
                secureTextEntry={!showPassword}
                autoCapitalize="none"
                autoCorrect={false}
              />
              <TouchableOpacity
                onPress={() => setShowPassword(!showPassword)}
                style={styles.eyeIcon}
              >
                <Icon
                  name={showPassword ? 'eye-off-outline' : 'eye-outline'}
                  size={20}
                  color={c.textMuted}
                />
              </TouchableOpacity>
            </View>
            {errors.password && (
              <Text style={[styles.errorText, { color: c.danger }]}>{errors.password}</Text>
            )}
          </View>

          {/* Forgot Password */}
          <TouchableOpacity style={styles.forgotPassword}>
            <Text style={[styles.forgotPasswordText, { color: c.primary }]}>
              Forgot Password?
            </Text>
          </TouchableOpacity>

          {/* Login Button */}
          <TouchableOpacity
            style={[
              styles.loginButton,
              { backgroundColor: c.primary },
              isLoading && styles.loginButtonDisabled
            ]}
            onPress={handleLogin}
            disabled={isLoading}
          >
            {isLoading ? (
              <ActivityIndicator color={c.background} size="small" />
            ) : (
              <Text style={[styles.loginButtonText, { color: c.background }]}>
                Sign In
              </Text>
            )}
          </TouchableOpacity>

          {/* Divider */}
          <View style={styles.divider}>
            <View style={[styles.dividerLine, { backgroundColor: c.border }]} />
            <Text style={[styles.dividerText, { color: c.textMuted }]}>or</Text>
            <View style={[styles.dividerLine, { backgroundColor: c.border }]} />
          </View>

          {/* Social Login */}
          <View style={styles.socialButtons}>
            <TouchableOpacity style={[styles.socialButton, { borderColor: c.border }]}>
              <Icon name="google" size={20} color="#DB4437" />
              <Text style={[styles.socialButtonText, { color: c.text }]}>
                Continue with Google
              </Text>
            </TouchableOpacity>
            
            <TouchableOpacity style={[styles.socialButton, { borderColor: c.border }]}>
              <Icon name="microsoft" size={20} color="#0078D4" />
              <Text style={[styles.socialButtonText, { color: c.text }]}>
                Continue with Microsoft
              </Text>
            </TouchableOpacity>
            
            <TouchableOpacity style={[styles.socialButton, { borderColor: c.border }]}>
              <Icon name="apple" size={20} color={c.text} />
              <Text style={[styles.socialButtonText, { color: c.text }]}>
                Continue with Apple
              </Text>
            </TouchableOpacity>
          </View>

          {/* Sign Up Link */}
          <View style={styles.signupContainer}>
            <Text style={[styles.signupText, { color: c.textMuted }]}>
              Don't have an account?{' '}
            </Text>
            <TouchableOpacity onPress={onNavigateToSignup}>
              <Text style={[styles.signupLink, { color: c.primary }]}>Sign Up</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollContainer: {
    flexGrow: 1,
    paddingHorizontal: 24,
    paddingTop: 60,
    paddingBottom: 40,
  },
  header: {
    alignItems: 'center',
    marginBottom: 40,
  },
  logoContainer: {
    width: 80,
    height: 80,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    textAlign: 'center',
    lineHeight: 24,
  },
  form: {
    flex: 1,
  },
  inputContainer: {
    marginBottom: 20,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 8,
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 16,
    gap: 12,
  },
  input: {
    flex: 1,
    fontSize: 16,
  },
  eyeIcon: {
    padding: 4,
  },
  errorText: {
    fontSize: 12,
    marginTop: 4,
  },
  forgotPassword: {
    alignSelf: 'flex-end',
    marginBottom: 24,
  },
  forgotPasswordText: {
    fontSize: 14,
    fontWeight: '600',
  },
  loginButton: {
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
    marginBottom: 24,
  },
  loginButtonDisabled: {
    opacity: 0.6,
  },
  loginButtonText: {
    fontSize: 16,
    fontWeight: '700',
  },
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 24,
  },
  dividerLine: {
    flex: 1,
    height: 1,
  },
  dividerText: {
    marginHorizontal: 16,
    fontSize: 14,
  },
  socialButtons: {
    gap: 12,
    marginBottom: 32,
  },
  socialButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderRadius: 12,
    paddingVertical: 16,
    gap: 12,
  },
  socialButtonText: {
    fontSize: 16,
    fontWeight: '600',
  },
  signupContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  signupText: {
    fontSize: 14,
  },
  signupLink: {
    fontSize: 14,
    fontWeight: '600',
  },
});

export default LoginScreen;
