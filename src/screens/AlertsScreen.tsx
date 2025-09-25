import React from 'react'
import { View, Text, StyleSheet, ScrollView, useColorScheme } from 'react-native'
import { Header, Card, MetricPill } from '../components/index'
import { colors } from '../theme/colors'

const AlertsScreen: React.FC = () => {
  const colorScheme = useColorScheme()
  const isDarkMode = true // Force dark mode
  const c = colors[isDarkMode ? 'dark' : 'light']

  return (
    <View style={[styles.container, { backgroundColor: c.background }]}> 
      <Header 
        title="Alerts" 
        subtitle="Stay informed with real-time notifications"
        rightIconName="bell-outline" 
      />
      <ScrollView contentContainerStyle={styles.content}>
        <Card variant="elevated">
          <View style={styles.alertHeader}>
            <Text style={[styles.alertTitle, { color: c.text }]}>Market Alert: S&P 500 down 2%</Text>
            <MetricPill label="Critical" value="High" variant="danger" />
          </View>
          <Text style={[styles.alertDescription, { color: c.textMuted }]}>
            Inflation fears triggered risk-off rotation. Consider defensive positioning.
          </Text>
          <Text style={[styles.alertTime, { color: c.textMuted }]}>2 minutes ago</Text>
        </Card>

        <Card>
          <View style={styles.alertHeader}>
            <Text style={[styles.alertTitle, { color: c.text }]}>Data Quality Alert</Text>
            <MetricPill label="Warning" value="Medium" variant="warning" />
          </View>
          <Text style={[styles.alertDescription, { color: c.textMuted }]}>
            Gold price feed delayed. Manual verification recommended.
          </Text>
          <Text style={[styles.alertTime, { color: c.textMuted }]}>15 minutes ago</Text>
        </Card>

        <Card>
          <View style={styles.alertHeader}>
            <Text style={[styles.alertTitle, { color: c.text }]}>Portfolio Rebalancing</Text>
            <MetricPill label="Info" value="Low" variant="default" />
          </View>
          <Text style={[styles.alertDescription, { color: c.textMuted }]}>
            Target allocation reached. Consider rebalancing your portfolio.
          </Text>
          <Text style={[styles.alertTime, { color: c.textMuted }]}>1 hour ago</Text>
        </Card>
      </ScrollView>
    </View>
  )
}

export default AlertsScreen

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: 16, gap: 16 },
  alertHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  alertTitle: {
    fontSize: 16,
    fontWeight: '700',
    flex: 1,
    marginRight: 12,
  },
  alertDescription: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 8,
  },
  alertTime: {
    fontSize: 12,
    fontWeight: '500',
  },
})


