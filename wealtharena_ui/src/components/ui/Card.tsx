import React, { ReactNode } from 'react'
import { View, StyleSheet, useColorScheme } from 'react-native'
import { colors } from '../../theme/colors'

type CardProps = {
  children: ReactNode
  padding?: number
  variant?: 'default' | 'elevated' | 'outlined'
}

export const Card: React.FC<CardProps> = ({ 
  children, 
  padding = 16, 
  variant = 'default' 
}) => {
  const colorScheme = useColorScheme()
  const isDarkMode = true // Force dark mode
  const c = colors[isDarkMode ? 'dark' : 'light']

  const getCardStyle = () => {
    switch (variant) {
      case 'elevated':
        return {
          backgroundColor: c.surface,
          borderWidth: 0,
          shadowColor: c.shadow.medium,
          shadowOffset: { width: 0, height: 4 },
          shadowOpacity: 1,
          shadowRadius: 12,
          elevation: 8,
        }
      case 'outlined':
        return {
          backgroundColor: c.surface,
          borderWidth: 1,
          borderColor: c.border,
        }
      default:
        return {
          backgroundColor: c.surface,
          borderWidth: StyleSheet.hairlineWidth,
          borderColor: c.border,
        }
    }
  }

  return (
    <View style={[styles.card, getCardStyle(), { padding }]}>
      {children}
    </View>
  )
}

const styles = StyleSheet.create({
  card: {
    borderRadius: 16,
    marginVertical: 4,
  },
})


