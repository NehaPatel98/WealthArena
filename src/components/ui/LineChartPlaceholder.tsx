import React from 'react'
import { View, StyleSheet, Text, useColorScheme } from 'react-native'
import Icon from 'react-native-vector-icons/MaterialCommunityIcons'
import { colors } from '../../theme/colors'

export const LineChartPlaceholder: React.FC = () => {
  const colorScheme = useColorScheme()
  const isDarkMode = true // Force dark mode
  const c = colors[isDarkMode ? 'dark' : 'light']

  return (
    <View style={[
      styles.chart, 
      { 
        backgroundColor: c.surfaceSecondary, 
        borderColor: c.border,
        shadowColor: c.shadow.small,
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 1,
        shadowRadius: 4,
        elevation: 2,
      }
    ]}>
      <View style={styles.chartContent}>
        <Icon name="chart-line" size={32} color={c.primary} />
        <Text style={[styles.chartText, { color: c.textMuted }]}>Portfolio Performance</Text>
        <Text style={[styles.chartSubtext, { color: c.textMuted }]}>Interactive chart coming soon</Text>
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  chart: {
    height: 160,
    borderRadius: 12,
    borderWidth: 1,
    marginTop: 8,
  },
  chartContent: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  chartText: {
    fontSize: 16,
    fontWeight: '600',
  },
  chartSubtext: {
    fontSize: 12,
    fontWeight: '400',
  },
})


