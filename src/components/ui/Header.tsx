import React from 'react'
import { View, Text, StyleSheet, TouchableOpacity, useColorScheme } from 'react-native'
import Icon from 'react-native-vector-icons/MaterialCommunityIcons'
import { colors } from '../../theme/colors'

type HeaderProps = {
  title: string
  rightIconName?: string
  onRightIconPress?: () => void
  subtitle?: string
}

export const Header: React.FC<HeaderProps> = ({ 
  title, 
  rightIconName, 
  onRightIconPress,
  subtitle 
}) => {
  const colorScheme = useColorScheme()
  const isDarkMode = true // Force dark mode
  const c = colors[isDarkMode ? 'dark' : 'light']

  return (
    <View style={[
      styles.container, 
      { 
        backgroundColor: c.background, 
        borderBottomColor: c.border,
        shadowColor: c.shadow.small,
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 1,
        shadowRadius: 4,
        elevation: 4,
      }
    ]}>
      <View style={styles.titleContainer}>
        <Text style={[styles.title, { color: c.text }]}>{title}</Text>
        {subtitle && (
          <Text style={[styles.subtitle, { color: c.textMuted }]}>{subtitle}</Text>
        )}
      </View>
      {rightIconName && (
        <TouchableOpacity 
          onPress={onRightIconPress} 
          style={[
            styles.rightIcon,
            {
              backgroundColor: c.surface,
              borderColor: c.border,
            }
          ]}
        >
          <Icon name={rightIconName} size={20} color={c.primary} />
        </TouchableOpacity>
      )}
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 20,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  titleContainer: {
    flex: 1,
  },
  title: {
    fontSize: 24,
    fontWeight: '800',
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: 14,
    fontWeight: '500',
    marginTop: 2,
  },
  rightIcon: {
    padding: 12,
    borderRadius: 12,
    borderWidth: 1,
  },
})


