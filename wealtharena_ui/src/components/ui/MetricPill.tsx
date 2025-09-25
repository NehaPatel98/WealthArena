import React from 'react'
import { View, Text, StyleSheet, useColorScheme } from 'react-native'
import Icon from 'react-native-vector-icons/MaterialCommunityIcons'
import { colors } from '../../theme/colors'

type MetricPillProps = {
  label: string
  value: string
  trend?: 'up' | 'down' | 'flat'
  variant?: 'default' | 'success' | 'warning' | 'danger'
}

export const MetricPill: React.FC<MetricPillProps> = ({ 
  label, 
  value, 
  trend = 'flat',
  variant = 'default'
}) => {
  const colorScheme = useColorScheme()
  const isDarkMode = true // Force dark mode
  const c = colors[isDarkMode ? 'dark' : 'light']

  const getTrendIcon = () => {
    switch (trend) {
      case 'up': return 'trending-up'
      case 'down': return 'trending-down'
      default: return 'trending-neutral'
    }
  }

  const getTrendColor = () => {
    switch (trend) {
      case 'up': return c.success
      case 'down': return c.danger
      default: return c.textMuted
    }
  }

  const getVariantStyle = () => {
    switch (variant) {
      case 'success':
        return {
          backgroundColor: c.success + '20',
          borderColor: c.success,
        }
      case 'warning':
        return {
          backgroundColor: c.warning + '20',
          borderColor: c.warning,
        }
      case 'danger':
        return {
          backgroundColor: c.danger + '20',
          borderColor: c.danger,
        }
      default:
        return {
          backgroundColor: c.surface,
          borderColor: c.border,
        }
    }
  }

  return (
    <View style={[
      styles.container, 
      getVariantStyle(),
      {
        shadowColor: c.shadow.small,
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 1,
        shadowRadius: 4,
        elevation: 2,
      }
    ]}> 
      <Text style={[styles.label, { color: c.textMuted }]}>{label}</Text>
      <View style={styles.valueContainer}>
        <Text style={[styles.value, { color: c.text }]}>{value}</Text>
        {trend !== 'flat' && (
          <Icon 
            name={getTrendIcon()} 
            size={12} 
            color={getTrendColor()} 
            style={styles.trendIcon}
          />
        )}
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    borderRadius: 20,
    borderWidth: 1,
    paddingHorizontal: 16,
    paddingVertical: 8,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    minWidth: 80,
  },
  label: { 
    fontSize: 11, 
    fontWeight: '500',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  valueContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  value: { 
    fontSize: 13, 
    fontWeight: '700',
  },
  trendIcon: {
    marginLeft: 2,
  },
})


