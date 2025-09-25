import React from 'react'
import { View, Text, StyleSheet, useColorScheme } from 'react-native'
import Icon from 'react-native-vector-icons/MaterialCommunityIcons'
import { colors } from '../../theme/colors'

type InsightItemProps = {
  icon: string
  title: string
  detail: string
  badge?: string
  variant?: 'default' | 'success' | 'warning' | 'danger'
}

export const InsightItem: React.FC<InsightItemProps> = ({ 
  icon, 
  title, 
  detail, 
  badge,
  variant = 'default'
}) => {
  const colorScheme = useColorScheme()
  const isDarkMode = true // Force dark mode
  const c = colors[isDarkMode ? 'dark' : 'light']

  const getVariantStyle = () => {
    switch (variant) {
      case 'success':
        return {
          iconColor: c.success,
          borderColor: c.success + '30',
        }
      case 'warning':
        return {
          iconColor: c.warning,
          borderColor: c.warning + '30',
        }
      case 'danger':
        return {
          iconColor: c.danger,
          borderColor: c.danger + '30',
        }
      default:
        return {
          iconColor: c.primary,
          borderColor: c.border,
        }
    }
  }

  const variantStyle = getVariantStyle()

  return (
    <View style={[
      styles.row, 
      { 
        borderColor: variantStyle.borderColor,
        backgroundColor: c.surfaceSecondary,
        borderRadius: 12,
        marginVertical: 4,
        paddingHorizontal: 12,
        shadowColor: c.shadow.small,
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 1,
        shadowRadius: 2,
        elevation: 1,
      }
    ]}> 
      <View style={[
        styles.iconContainer,
        { backgroundColor: variantStyle.iconColor + '20' }
      ]}>
        <Icon name={icon} size={16} color={variantStyle.iconColor} />
      </View>
      <View style={{ flex: 1, marginLeft: 12 }}>
        <Text style={[styles.title, { color: c.text }]} numberOfLines={1}>{title}</Text>
        <Text style={[styles.detail, { color: c.textMuted }]} numberOfLines={1}>{detail}</Text>
      </View>
      {badge && (
        <View style={[
          styles.badge, 
          { 
            backgroundColor: variantStyle.iconColor + '20',
            borderColor: variantStyle.iconColor,
          }
        ]}>
          <Text style={[styles.badgeText, { color: variantStyle.iconColor }]}>{badge}</Text>
        </View>
      )}
    </View>
  )
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    borderWidth: 1,
  },
  iconContainer: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: { 
    fontSize: 14, 
    fontWeight: '600',
    marginBottom: 2,
  },
  detail: { 
    fontSize: 12, 
    fontWeight: '400',
  },
  badge: { 
    borderRadius: 12, 
    paddingHorizontal: 8, 
    paddingVertical: 4, 
    marginLeft: 8,
    borderWidth: 1,
  },
  badgeText: {
    fontSize: 10,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
})


