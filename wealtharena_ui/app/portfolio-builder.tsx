import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { Stack } from 'expo-router';
import { Briefcase, ChevronRight, Check } from 'lucide-react-native';
import Colors from '@/constants/colors';

const STEPS = ['Constraints', 'Suggestions', 'Stress Test', 'Review'];

const RISK_LEVELS = ['Conservative', 'Moderate', 'Aggressive'];
const SECTORS = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer', 'Industrial'];

export default function PortfolioBuilderScreen() {
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedRisk, setSelectedRisk] = useState<string>('Moderate');
  const [selectedSectors, setSelectedSectors] = useState<string[]>([]);

  const toggleSector = (sector: string) => {
    setSelectedSectors((prev) =>
      prev.includes(sector) ? prev.filter((s) => s !== sector) : [...prev, sector]
    );
  };

  return (
    <View style={styles.container}>
      <Stack.Screen
        options={{
          title: 'Portfolio Builder',
          headerStyle: { backgroundColor: Colors.background },
          headerTintColor: Colors.text,
        }}
      />

      <View style={styles.stepsContainer}>
        {STEPS.map((step, index) => (
          <View key={step} style={styles.stepItem}>
            <View
              style={[
                styles.stepCircle,
                index <= currentStep && styles.stepCircleActive,
                index < currentStep && styles.stepCircleCompleted,
              ]}
            >
              {index < currentStep ? (
                <Check size={16} color={Colors.text} />
              ) : (
                <Text
                  style={[
                    styles.stepNumber,
                    index <= currentStep && styles.stepNumberActive,
                  ]}
                >
                  {index + 1}
                </Text>
              )}
            </View>
            <Text
              style={[
                styles.stepLabel,
                index <= currentStep && styles.stepLabelActive,
              ]}
            >
              {step}
            </Text>
          </View>
        ))}
      </View>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {currentStep === 0 && (
          <View style={styles.stepContent}>
            <View style={styles.headerSection}>
              <Briefcase size={32} color={Colors.secondary} />
              <Text style={styles.stepTitle}>Define Your Constraints</Text>
              <Text style={styles.stepDescription}>
                Set your risk tolerance and sector preferences
              </Text>
            </View>

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Risk Tolerance</Text>
              <View style={styles.optionsGrid}>
                {RISK_LEVELS.map((risk) => (
                  <TouchableOpacity
                    key={risk}
                    style={[
                      styles.optionCard,
                      selectedRisk === risk && styles.optionCardSelected,
                    ]}
                    onPress={() => setSelectedRisk(risk)}
                  >
                    <Text
                      style={[
                        styles.optionText,
                        selectedRisk === risk && styles.optionTextSelected,
                      ]}
                    >
                      {risk}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Sector Preferences</Text>
              <Text style={styles.sectionSubtitle}>Select sectors to include</Text>
              <View style={styles.sectorsGrid}>
                {SECTORS.map((sector) => (
                  <TouchableOpacity
                    key={sector}
                    style={[
                      styles.sectorChip,
                      selectedSectors.includes(sector) && styles.sectorChipSelected,
                    ]}
                    onPress={() => toggleSector(sector)}
                  >
                    {selectedSectors.includes(sector) && (
                      <Check size={16} color={Colors.text} />
                    )}
                    <Text
                      style={[
                        styles.sectorText,
                        selectedSectors.includes(sector) && styles.sectorTextSelected,
                      ]}
                    >
                      {sector}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
          </View>
        )}

        {currentStep === 1 && (
          <View style={styles.stepContent}>
            <View style={styles.headerSection}>
              <Text style={styles.stepTitle}>Suggested Portfolio</Text>
              <Text style={styles.stepDescription}>
                Based on your preferences, here&apos;s our recommendation
              </Text>
            </View>

            <View style={styles.allocationCard}>
              <Text style={styles.allocationTitle}>Asset Allocation</Text>
              {[
                { name: 'US Stocks', percent: 40, color: Colors.secondary },
                { name: 'International Stocks', percent: 25, color: Colors.accent },
                { name: 'Bonds', percent: 20, color: Colors.gold },
                { name: 'Real Estate', percent: 10, color: Colors.warning },
                { name: 'Cash', percent: 5, color: Colors.textMuted },
              ].map((asset) => (
                <View key={asset.name} style={styles.allocationRow}>
                  <View style={styles.allocationLeft}>
                    <View style={[styles.allocationDot, { backgroundColor: asset.color }]} />
                    <Text style={styles.allocationName}>{asset.name}</Text>
                  </View>
                  <Text style={styles.allocationPercent}>{asset.percent}%</Text>
                </View>
              ))}
            </View>

            <View style={styles.metricsCard}>
              <Text style={styles.metricsTitle}>Expected Metrics</Text>
              <View style={styles.metricsGrid}>
                <View style={styles.metricItem}>
                  <Text style={styles.metricLabel}>Expected Return</Text>
                  <Text style={styles.metricValue}>8.5%</Text>
                </View>
                <View style={styles.metricItem}>
                  <Text style={styles.metricLabel}>Volatility</Text>
                  <Text style={styles.metricValue}>12.3%</Text>
                </View>
                <View style={styles.metricItem}>
                  <Text style={styles.metricLabel}>Sharpe Ratio</Text>
                  <Text style={styles.metricValue}>1.42</Text>
                </View>
              </View>
            </View>
          </View>
        )}

        {currentStep === 2 && (
          <View style={styles.stepContent}>
            <View style={styles.headerSection}>
              <Text style={styles.stepTitle}>Stress Test Results</Text>
              <Text style={styles.stepDescription}>
                How your portfolio performs under different scenarios
              </Text>
            </View>

            {[
              { scenario: 'Market Crash (-30%)', impact: '-18.5%', color: Colors.danger },
              { scenario: 'Interest Rate Spike', impact: '-8.2%', color: Colors.warning },
              { scenario: 'Inflation Surge', impact: '-5.1%', color: Colors.warning },
              { scenario: 'Bull Market (+40%)', impact: '+28.3%', color: Colors.accent },
            ].map((test) => (
              <View key={test.scenario} style={styles.stressCard}>
                <Text style={styles.stressScenario}>{test.scenario}</Text>
                <Text style={[styles.stressImpact, { color: test.color }]}>
                  {test.impact}
                </Text>
              </View>
            ))}
          </View>
        )}

        {currentStep === 3 && (
          <View style={styles.stepContent}>
            <View style={styles.headerSection}>
              <Text style={styles.stepTitle}>Review & Save</Text>
              <Text style={styles.stepDescription}>
                Your portfolio is ready to be saved
              </Text>
            </View>

            <View style={styles.summaryCard}>
              <Text style={styles.summaryTitle}>Portfolio Summary</Text>
              <View style={styles.summaryRow}>
                <Text style={styles.summaryLabel}>Risk Level</Text>
                <Text style={styles.summaryValue}>{selectedRisk}</Text>
              </View>
              <View style={styles.summaryRow}>
                <Text style={styles.summaryLabel}>Sectors</Text>
                <Text style={styles.summaryValue}>
                  {selectedSectors.length || 'All'}
                </Text>
              </View>
              <View style={styles.summaryRow}>
                <Text style={styles.summaryLabel}>Expected Return</Text>
                <Text style={[styles.summaryValue, { color: Colors.accent }]}>
                  8.5%
                </Text>
              </View>
            </View>

            <TouchableOpacity style={styles.saveButton}>
              <Text style={styles.saveButtonText}>Save Portfolio</Text>
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>

      <View style={styles.navigation}>
        {currentStep > 0 && (
          <TouchableOpacity
            style={styles.navButton}
            onPress={() => setCurrentStep((prev) => prev - 1)}
          >
            <Text style={styles.navButtonText}>Back</Text>
          </TouchableOpacity>
        )}
        {currentStep < STEPS.length - 1 && (
          <TouchableOpacity
            style={[styles.navButton, styles.navButtonPrimary]}
            onPress={() => setCurrentStep((prev) => prev + 1)}
          >
            <Text style={styles.navButtonTextPrimary}>Next</Text>
            <ChevronRight size={20} color={Colors.text} />
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  stepsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 24,
    paddingVertical: 20,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  stepItem: {
    alignItems: 'center',
    gap: 8,
    flex: 1,
  },
  stepCircle: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: Colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: Colors.border,
  },
  stepCircleActive: {
    borderColor: Colors.secondary,
    backgroundColor: Colors.surface,
  },
  stepCircleCompleted: {
    backgroundColor: Colors.secondary,
    borderColor: Colors.secondary,
  },
  stepNumber: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.textMuted,
  },
  stepNumberActive: {
    color: Colors.secondary,
  },
  stepLabel: {
    fontSize: 11,
    color: Colors.textMuted,
    textAlign: 'center',
  },
  stepLabelActive: {
    color: Colors.text,
    fontWeight: '600' as const,
  },
  content: {
    flex: 1,
  },
  stepContent: {
    padding: 24,
    gap: 24,
  },
  headerSection: {
    alignItems: 'center',
    gap: 12,
  },
  stepTitle: {
    fontSize: 24,
    fontWeight: '700' as const,
    color: Colors.text,
    textAlign: 'center',
  },
  stepDescription: {
    fontSize: 14,
    color: Colors.textSecondary,
    textAlign: 'center',
  },
  section: {
    gap: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  sectionSubtitle: {
    fontSize: 14,
    color: Colors.textSecondary,
  },
  optionsGrid: {
    flexDirection: 'row',
    gap: 12,
  },
  optionCard: {
    flex: 1,
    backgroundColor: Colors.surface,
    padding: 16,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: Colors.border,
    alignItems: 'center',
  },
  optionCardSelected: {
    borderColor: Colors.secondary,
    backgroundColor: Colors.secondary + '20',
  },
  optionText: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.textSecondary,
  },
  optionTextSelected: {
    color: Colors.text,
  },
  sectorsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  sectorChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 16,
    paddingVertical: 10,
    backgroundColor: Colors.surface,
    borderRadius: 20,
    borderWidth: 2,
    borderColor: Colors.border,
  },
  sectorChipSelected: {
    borderColor: Colors.secondary,
    backgroundColor: Colors.secondary + '20',
  },
  sectorText: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.textSecondary,
  },
  sectorTextSelected: {
    color: Colors.text,
  },
  allocationCard: {
    backgroundColor: Colors.surface,
    padding: 20,
    borderRadius: 16,
    gap: 16,
  },
  allocationTitle: {
    fontSize: 18,
    fontWeight: '600' as const,
    color: Colors.text,
    marginBottom: 8,
  },
  allocationRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  allocationLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  allocationDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  allocationName: {
    fontSize: 14,
    color: Colors.text,
  },
  allocationPercent: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  metricsCard: {
    backgroundColor: Colors.surface,
    padding: 20,
    borderRadius: 16,
    gap: 16,
  },
  metricsTitle: {
    fontSize: 18,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  metricsGrid: {
    flexDirection: 'row',
    gap: 16,
  },
  metricItem: {
    flex: 1,
    gap: 8,
  },
  metricLabel: {
    fontSize: 12,
    color: Colors.textSecondary,
  },
  metricValue: {
    fontSize: 20,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  stressCard: {
    backgroundColor: Colors.surface,
    padding: 20,
    borderRadius: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  stressScenario: {
    fontSize: 16,
    color: Colors.text,
    fontWeight: '500' as const,
  },
  stressImpact: {
    fontSize: 18,
    fontWeight: '700' as const,
  },
  summaryCard: {
    backgroundColor: Colors.surface,
    padding: 20,
    borderRadius: 16,
    gap: 16,
  },
  summaryTitle: {
    fontSize: 18,
    fontWeight: '600' as const,
    color: Colors.text,
    marginBottom: 8,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  summaryLabel: {
    fontSize: 14,
    color: Colors.textSecondary,
  },
  summaryValue: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  saveButton: {
    backgroundColor: Colors.secondary,
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  saveButtonText: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  navigation: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 24,
    borderTopWidth: 1,
    borderTopColor: Colors.border,
    gap: 12,
  },
  navButton: {
    flex: 1,
    padding: 16,
    borderRadius: 12,
    backgroundColor: Colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  navButtonPrimary: {
    backgroundColor: Colors.secondary,
    flexDirection: 'row',
    gap: 8,
  },
  navButtonText: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  navButtonTextPrimary: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text,
  },
});
