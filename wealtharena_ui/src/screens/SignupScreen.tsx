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

interface SignupScreenProps {
  onSignup: (email: string, password: string, confirmPassword: string, fullName: string) => Promise<void>;
  onNavigateToLogin: () => void;
}

const SignupScreen: React.FC<SignupScreenProps> = ({ onSignup, onNavigateToLogin }) => {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<{
    fullName?: string;
    email?: string;
    password?: string;
    confirmPassword?: string;
  }>({});

  const isDarkMode = true;
  const c = colors[isDarkMode ? 'dark' : 'light'];

  const validateForm = () => {
    const newErrors: {
      fullName?: string;
      email?: string;
      password?: string;
      confirmPassword?: string;
    } = {};

    if (!fullName.trim()) {
      newErrors.fullName = 'Full name is required';
    } else if (fullName.trim().length < 2) {
      newErrors.fullName = 'Full name must be at least 2 characters';
    }

    if (!email.trim()) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(email)) {
      newErrors.email = 'Please enter a valid email';
    }

    if (!password.trim()) {
      newErrors.password = 'Password is required';
    } else if (password.length < 8) {
      newErrors.password = 'Password must be at least 8 characters';
    } else if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/.test(password)) {
      newErrors.password = 'Password must contain uppercase, lowercase, and number';
    }

    if (!confirmPassword.trim()) {
      newErrors.confirmPassword = 'Please confirm your password';
    } else if (password !== confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSignup = async () => {
    if (!validateForm()) return;

    setIsLoading(true);
    try {
      await onSignup(fullName, email, password, confirmPassword);
    } catch (error) {
      Alert.alert('Signup Failed', 'Unable to create account. Please try again.');
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
          <Text style={[styles.title, { color: c.text }]}>Create Account</Text>
          <Text style={[styles.subtitle, { color: c.textMuted }]}>
            Join WealthArena and start your investment journey
          </Text>
        </View>

        {/* Form */}
        <View style={styles.form}>
          {/* Full Name Input */}
          <View style={styles.inputContainer}>
            <Text style={[styles.label, { color: c.text }]}>Full Name</Text>
            <View style={[styles.inputWrapper, { borderColor: errors.fullName ? c.danger : c.border }]}>
              <Icon name="account-outline" size={20} color={c.textMuted} />
              <TextInput
                style={[styles.input, { color: c.text }]}
                placeholder="Enter your full name"
                placeholderTextColor={c.textMuted}
                value={fullName}
                onChangeText={setFullName}
                autoCapitalize="words"
                autoCorrect={false}
              />
            </View>
            {errors.fullName && (
              <Text style={[styles.errorText, { color: c.danger }]}>{errors.fullName}</Text>
            )}
          </View>

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
                placeholder="Create a strong password"
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

          {/* Confirm Password Input */}
          <View style={styles.inputContainer}>
            <Text style={[styles.label, { color: c.text }]}>Confirm Password</Text>
            <View style={[styles.inputWrapper, { borderColor: errors.confirmPassword ? c.danger : c.border }]}>
              <Icon name="lock-check-outline" size={20} color={c.textMuted} />
              <TextInput
                style={[styles.input, { color: c.text }]}
                placeholder="Confirm your password"
                placeholderTextColor={c.textMuted}
                value={confirmPassword}
                onChangeText={setConfirmPassword}
                secureTextEntry={!showConfirmPassword}
                autoCapitalize="none"
                autoCorrect={false}
              />
              <TouchableOpacity
                onPress={() => setShowConfirmPassword(!showConfirmPassword)}
                style={styles.eyeIcon}
              >
                <Icon
                  name={showConfirmPassword ? 'eye-off-outline' : 'eye-outline'}
                  size={20}
                  color={c.textMuted}
                />
              </TouchableOpacity>
            </View>
            {errors.confirmPassword && (
              <Text style={[styles.errorText, { color: c.danger }]}>{errors.confirmPassword}</Text>
            )}
          </View>

          {/* Terms and Conditions */}
          <View style={styles.termsContainer}>
            <Text style={[styles.termsText, { color: c.textMuted }]}>
              By creating an account, you agree to our{' '}
            </Text>
            <TouchableOpacity>
              <Text style={[styles.termsLink, { color: c.primary }]}>Terms of Service</Text>
            </TouchableOpacity>
            <Text style={[styles.termsText, { color: c.textMuted }]}> and </Text>
            <TouchableOpacity>
              <Text style={[styles.termsLink, { color: c.primary }]}>Privacy Policy</Text>
            </TouchableOpacity>
          </View>

          {/* Signup Button */}
          <TouchableOpacity
            style={[
              styles.signupButton,
              { backgroundColor: c.primary },
              isLoading && styles.signupButtonDisabled
            ]}
            onPress={handleSignup}
            disabled={isLoading}
          >
            {isLoading ? (
              <ActivityIndicator color={c.background} size="small" />
            ) : (
              <Text style={[styles.signupButtonText, { color: c.background }]}>
                Create Account
              </Text>
            )}
          </TouchableOpacity>

          {/* Divider */}
          <View style={styles.divider}>
            <View style={[styles.dividerLine, { backgroundColor: c.border }]} />
            <Text style={[styles.dividerText, { color: c.textMuted }]}>or</Text>
            <View style={[styles.dividerLine, { backgroundColor: c.border }]} />
          </View>

          {/* Social Signup */}
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

          {/* Login Link */}
          <View style={styles.loginContainer}>
            <Text style={[styles.loginText, { color: c.textMuted }]}>
              Already have an account?{' '}
            </Text>
            <TouchableOpacity onPress={onNavigateToLogin}>
              <Text style={[styles.loginLink, { color: c.primary }]}>Sign In</Text>
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
    marginBottom: 32,
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
  termsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    marginBottom: 24,
  },
  termsText: {
    fontSize: 14,
  },
  termsLink: {
    fontSize: 14,
    fontWeight: '600',
  },
  signupButton: {
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
    marginBottom: 24,
  },
  signupButtonDisabled: {
    opacity: 0.6,
  },
  signupButtonText: {
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
  loginContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loginText: {
    fontSize: 14,
  },
  loginLink: {
    fontSize: 14,
    fontWeight: '600',
  },
});

export default SignupScreen;
