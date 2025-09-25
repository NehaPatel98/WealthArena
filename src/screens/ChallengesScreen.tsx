import React from 'react'
import { ScrollView, View, Text, StyleSheet, useColorScheme } from 'react-native'
import { Header, Card, LineChartPlaceholder } from '../components/index'
import { colors } from '../theme/colors'

const ChallengesScreen: React.FC = () => {
  const colorScheme = useColorScheme()
  const isDarkMode = true // Force dark mode
  const c = colors[isDarkMode ? 'dark' : 'light']

  return (
    <View style={[styles.container, { backgroundColor: c.background }]}> 
      <Header title="Historical Challenges" rightIconName="chevron-right" />
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.grid2}> 
          {Array.from({ length: 6 }).map((_, i) => (
            <Card key={i}>
              <Text style={[styles.cardTitle, { color: c.text }]}>Episode {i + 1}</Text>
              <LineChartPlaceholder />
              <Text style={{ color: c.text, marginTop: 8, fontWeight: '600' }}>$ {1000 + i * 120}</Text>
              <Text style={{ color: c.textMuted, marginTop: 2 }}>Aug–Dec 2020</Text>
            </Card>
          ))}
        </View>
      </ScrollView>
    </View>
  )
}

export default ChallengesScreen

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: 16, gap: 16 },
  grid2: { flexDirection: 'row', flexWrap: 'wrap', gap: 12 },
  cardTitle: { fontSize: 13, fontWeight: '700', marginBottom: 8 },
})


