import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  Switch,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { colors } from '../theme/colors';

interface UserProfile {
  experience: 'beginner' | 'intermediate' | 'advanced';
  riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  investmentGoals: string[];
  timeHorizon: 'short' | 'medium' | 'long';
  suggestedPortfolio: string[];
}

interface PortfolioConstraint {
  id: string;
  name: string;
  type: 'risk' | 'esg' | 'sector' | 'geography' | 'size';
  value: string | number;
  enabled: boolean;
}

interface Asset {
  id: string;
  name: string;
  symbol: string;
  category: string;
  riskLevel: 'low' | 'medium' | 'high';
  expectedReturn: number;
  volatility: number;
  esgScore?: number;
  sector?: string;
  geography?: string;
}

interface PortfolioBuilderScreenProps {
  userProfile: UserProfile;
  onPortfolioCreate: (portfolio: any) => void;
  onBack: () => void;
}

const PortfolioBuilderScreen: React.FC<PortfolioBuilderScreenProps> = ({
  userProfile,
  onPortfolioCreate,
  onBack
}) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [constraints, setConstraints] = useState<PortfolioConstraint[]>([]);
  const [suggestedPortfolio, setSuggestedPortfolio] = useState<Asset[]>([]);
  const [stressTestResults, setStressTestResults] = useState<any>(null);
  const [portfolioName, setPortfolioName] = useState('');
  const [totalAllocation, setTotalAllocation] = useState(100);
  
  const isDarkMode = true;
  const c = colors[isDarkMode ? 'dark' : 'light'];

  useEffect(() => {
    initializeConstraints();
  }, [initializeConstraints]);

  const initializeConstraints = useCallback(() => {
    const defaultConstraints: PortfolioConstraint[] = [
      {
        id: 'max-risk',
        name: 'Maximum Risk Level',
        type: 'risk',
        value: userProfile.riskTolerance,
        enabled: true
      },
      {
        id: 'esg-minimum',
        name: 'ESG Score Minimum',
        type: 'esg',
        value: 6,
        enabled: false
      },
      {
        id: 'sector-limit',
        name: 'Sector Concentration Limit',
        type: 'sector',
        value: 30,
        enabled: false
      },
      {
        id: 'geography-diversification',
        name: 'Geographic Diversification',
        type: 'geography',
        value: 'global',
        enabled: false
      },
      {
        id: 'small-cap-limit',
        name: 'Small Cap Limit',
        type: 'size',
        value: 20,
        enabled: false
      }
    ];
    setConstraints(defaultConstraints);
  }, [userProfile]);

  const availableAssets: Asset[] = [
    // Low Risk Assets
    {
      id: 'spy',
      name: 'SPDR S&P 500 ETF',
      symbol: 'SPY',
      category: 'ETF',
      riskLevel: 'low',
      expectedReturn: 8.5,
      volatility: 15.2,
      esgScore: 7.2,
      sector: 'Diversified',
      geography: 'US'
    },
    {
      id: 'bnd',
      name: 'Vanguard Total Bond Market ETF',
      symbol: 'BND',
      category: 'Bond',
      riskLevel: 'low',
      expectedReturn: 4.2,
      volatility: 3.8,
      esgScore: 6.8,
      sector: 'Fixed Income',
      geography: 'US'
    },
    {
      id: 'vti',
      name: 'Vanguard Total Stock Market ETF',
      symbol: 'VTI',
      category: 'ETF',
      riskLevel: 'low',
      expectedReturn: 9.1,
      volatility: 16.1,
      esgScore: 7.5,
      sector: 'Diversified',
      geography: 'US'
    },
    // Medium Risk Assets
    {
      id: 'vxus',
      name: 'Vanguard Total International Stock ETF',
      symbol: 'VXUS',
      category: 'ETF',
      riskLevel: 'medium',
      expectedReturn: 7.8,
      volatility: 18.5,
      esgScore: 7.8,
      sector: 'Diversified',
      geography: 'International'
    },
    {
      id: 'vwo',
      name: 'Vanguard Emerging Markets ETF',
      symbol: 'VWO',
      category: 'ETF',
      riskLevel: 'medium',
      expectedReturn: 10.2,
      volatility: 22.3,
      esgScore: 6.5,
      sector: 'Diversified',
      geography: 'Emerging Markets'
    },
    {
      id: 'vnq',
      name: 'Vanguard Real Estate ETF',
      symbol: 'VNQ',
      category: 'REIT',
      riskLevel: 'medium',
      expectedReturn: 6.5,
      volatility: 19.8,
      esgScore: 6.2,
      sector: 'Real Estate',
      geography: 'US'
    },
    // High Risk Assets
    {
      id: 'arkk',
      name: 'ARK Innovation ETF',
      symbol: 'ARKK',
      category: 'ETF',
      riskLevel: 'high',
      expectedReturn: 12.5,
      volatility: 35.2,
      esgScore: 8.1,
      sector: 'Technology',
      geography: 'Global'
    },
    {
      id: 'qqq',
      name: 'Invesco QQQ Trust',
      symbol: 'QQQ',
      category: 'ETF',
      riskLevel: 'high',
      expectedReturn: 11.8,
      volatility: 28.5,
      esgScore: 7.9,
      sector: 'Technology',
      geography: 'US'
    }
  ];

  const handleConstraintToggle = (constraintId: string) => {
    setConstraints(prev => 
      prev.map(constraint => 
        constraint.id === constraintId 
          ? { ...constraint, enabled: !constraint.enabled }
          : constraint
      )
    );
  };

  const generatePortfolio = () => {
    const enabledConstraints = constraints.filter(constraint => constraint.enabled);
    const filteredAssets = availableAssets.filter(asset => {
      return enabledConstraints.every(constraint => {
        switch (constraint.type) {
          case 'risk':
            if (constraint.value === 'conservative') return asset.riskLevel === 'low';
            if (constraint.value === 'moderate') return ['low', 'medium'].includes(asset.riskLevel);
            return true; // aggressive allows all
          case 'esg':
            return asset.esgScore && asset.esgScore >= (constraint.value as number);
          default:
            return true;
        }
      });
    });

    // Generate portfolio based on user profile and constraints
    const portfolio = generateSuggestedPortfolio(filteredAssets, userProfile);
    setSuggestedPortfolio(portfolio);
    setCurrentStep(2);
  };

  const generateSuggestedPortfolio = (assets: Asset[], profile: UserProfile): Asset[] => {
    let portfolio: Asset[] = [];
    
    if (profile.experience === 'beginner' || profile.riskTolerance === 'conservative') {
      portfolio = [
        { ...assets.find(a => a.symbol === 'SPY')!, percentage: 60 },
        { ...assets.find(a => a.symbol === 'BND')!, percentage: 30 },
        { ...assets.find(a => a.symbol === 'VTI')!, percentage: 10 }
      ];
    } else if (profile.experience === 'intermediate' || profile.riskTolerance === 'moderate') {
      portfolio = [
        { ...assets.find(a => a.symbol === 'SPY')!, percentage: 40 },
        { ...assets.find(a => a.symbol === 'VXUS')!, percentage: 20 },
        { ...assets.find(a => a.symbol === 'BND')!, percentage: 25 },
        { ...assets.find(a => a.symbol === 'VNQ')!, percentage: 15 }
      ];
    } else {
      portfolio = [
        { ...assets.find(a => a.symbol === 'ARKK')!, percentage: 20 },
        { ...assets.find(a => a.symbol === 'QQQ')!, percentage: 30 },
        { ...assets.find(a => a.symbol === 'VXUS')!, percentage: 25 },
        { ...assets.find(a => a.symbol === 'BND')!, percentage: 15 },
        { ...assets.find(a => a.symbol === 'VWO')!, percentage: 10 }
      ];
    }
    
    return portfolio;
  };

  const runStressTest = () => {
    const stressScenarios = [
      { name: '2008 Financial Crisis', return: -37 },
      { name: 'COVID-19 Crash', return: -34 },
      { name: 'Dot-com Bubble', return: -49 },
      { name: 'Interest Rate Shock', return: -15 }
    ];
    
    setStressTestResults(stressScenarios);
    setCurrentStep(3);
  };

  const savePortfolio = () => {
    if (!portfolioName.trim()) {
      Alert.alert('Error', 'Please enter a portfolio name');
      return;
    }
    
    const portfolio = {
      name: portfolioName,
      assets: suggestedPortfolio,
      constraints: constraints.filter(constraint => constraint.enabled),
      stressTestResults,
      createdAt: new Date().toISOString()
    };
    
    onPortfolioCreate(portfolio);
  };

  const renderStep1 = () => (
    <View style={styles.stepContainer}>
      <Text style={[styles.stepTitle, { color: c.text }]}>Step 1: Define Constraints</Text>
      <Text style={[styles.stepDescription, { color: c.textMuted }]}>
        Set your investment constraints and preferences
      </Text>
      
      <View style={styles.constraintsList}>
        {constraints.map((constraint) => (
          <View key={constraint.id} style={[styles.constraintItem, { backgroundColor: c.surface }]}>
            <View style={styles.constraintInfo}>
              <Text style={[styles.constraintName, { color: c.text }]}>{constraint.name}</Text>
              <Text style={[styles.constraintValue, { color: c.textMuted }]}>
                {constraint.type === 'risk' ? constraint.value : `${constraint.value}${constraint.type === 'esg' ? '/10' : constraint.type === 'sector' || constraint.type === 'size' ? '%' : ''}`}
              </Text>
            </View>
            <Switch
              value={constraint.enabled}
              onValueChange={() => handleConstraintToggle(constraint.id)}
              trackColor={{ false: c.border, true: c.primary }}
              thumbColor={constraint.enabled ? c.background : c.textMuted}
            />
          </View>
        ))}
      </View>
      
      <TouchableOpacity 
        style={[styles.nextButton, { backgroundColor: c.primary }]}
        onPress={generatePortfolio}
      >
        <Text style={[styles.nextButtonText, { color: c.background }]}>Generate Portfolio</Text>
        <Icon name="arrow-right" size={20} color={c.background} />
      </TouchableOpacity>
    </View>
  );

  const renderStep2 = () => (
    <View style={styles.stepContainer}>
      <Text style={[styles.stepTitle, { color: c.text }]}>Step 2: Suggested Portfolio</Text>
      <Text style={[styles.stepDescription, { color: c.textMuted }]}>
        Based on your constraints and profile
      </Text>
      
      <View style={styles.portfolioContainer}>
        {suggestedPortfolio.map((asset, index) => (
          <View key={asset.id} style={[styles.assetItem, { backgroundColor: c.surface }]}>
            <View style={styles.assetInfo}>
              <Text style={[styles.assetName, { color: c.text }]}>{asset.name}</Text>
              <Text style={[styles.assetSymbol, { color: c.textMuted }]}>{asset.symbol}</Text>
              <Text style={[styles.assetCategory, { color: c.textMuted }]}>{asset.category}</Text>
            </View>
            <View style={styles.assetMetrics}>
              <Text style={[styles.assetPercentage, { color: c.primary }]}>{asset.percentage}%</Text>
              <Text style={[styles.assetReturn, { color: c.success }]}>
                {asset.expectedReturn}% expected return
              </Text>
              <Text style={[styles.assetRisk, { color: c.textMuted }]}>
                {asset.riskLevel} risk
              </Text>
            </View>
          </View>
        ))}
      </View>
      
      <View style={styles.allocationSummary}>
        <Text style={[styles.allocationTitle, { color: c.text }]}>Total Allocation: {totalAllocation}%</Text>
        {totalAllocation !== 100 && (
          <Text style={[styles.allocationWarning, { color: c.danger }]}>
            Portfolio allocation must equal 100%
          </Text>
        )}
      </View>
      
      <View style={styles.stepButtons}>
        <TouchableOpacity 
          style={[styles.stepBackButton, { borderColor: c.border }]}
          onPress={() => setCurrentStep(1)}
        >
          <Text style={[styles.backButtonText, { color: c.text }]}>Back</Text>
        </TouchableOpacity>
        <TouchableOpacity 
          style={[styles.nextButton, { backgroundColor: c.primary }]}
          onPress={runStressTest}
        >
          <Text style={[styles.nextButtonText, { color: c.background }]}>Run Stress Tests</Text>
          <Icon name="flask" size={20} color={c.background} />
        </TouchableOpacity>
      </View>
    </View>
  );

  const renderStep3 = () => (
    <View style={styles.stepContainer}>
      <Text style={[styles.stepTitle, { color: c.text }]}>Step 3: Stress Test Results</Text>
      <Text style={[styles.stepDescription, { color: c.textMuted }]}>
        How your portfolio would perform in historical market crashes
      </Text>
      
      <View style={styles.stressTestContainer}>
        {stressTestResults?.map((scenario, index) => (
          <View key={index} style={[styles.scenarioItem, { backgroundColor: c.surface }]}>
            <Text style={[styles.scenarioName, { color: c.text }]}>{scenario.name}</Text>
            <Text style={[
              styles.scenarioReturn, 
              { color: scenario.return < 0 ? c.danger : c.success }
            ]}>
              {scenario.return}%
            </Text>
          </View>
        ))}
      </View>
      
      <View style={styles.portfolioNameContainer}>
        <Text style={[styles.portfolioNameLabel, { color: c.text }]}>Portfolio Name</Text>
        <TextInput
          style={[styles.portfolioNameInput, { 
            backgroundColor: c.background, 
            borderColor: c.border,
            color: c.text 
          }]}
          placeholder="Enter portfolio name"
          placeholderTextColor={c.textMuted}
          value={portfolioName}
          onChangeText={setPortfolioName}
        />
      </View>
      
      <View style={styles.stepButtons}>
        <TouchableOpacity 
          style={[styles.stepBackButton, { borderColor: c.border }]}
          onPress={() => setCurrentStep(2)}
        >
          <Text style={[styles.backButtonText, { color: c.text }]}>Back</Text>
        </TouchableOpacity>
        <TouchableOpacity 
          style={[styles.saveButton, { backgroundColor: c.success }]}
          onPress={savePortfolio}
        >
          <Text style={[styles.saveButtonText, { color: c.background }]}>Save Portfolio</Text>
          <Icon name="check" size={20} color={c.background} />
        </TouchableOpacity>
      </View>
    </View>
  );

  return (
    <View style={[styles.container, { backgroundColor: c.background }]}>
      {/* Header */}
      <View style={[styles.header, { borderBottomColor: c.border }]}>
        <TouchableOpacity onPress={onBack} style={styles.backButton}>
          <Icon name="arrow-left" size={24} color={c.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: c.text }]}>Portfolio Builder</Text>
        <View style={styles.stepIndicator}>
          <Text style={[styles.stepIndicatorText, { color: c.textMuted }]}>
            Step {currentStep} of 3
          </Text>
        </View>
      </View>

      {/* Progress Bar */}
      <View style={[styles.progressBar, { backgroundColor: c.border }]}>
        <View 
          style={[
            styles.progressFill, 
            { 
              backgroundColor: c.primary, 
              width: `${(currentStep / 3) * 100}%` 
            }
          ]} 
        />
      </View>

      {/* Content */}
      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {currentStep === 1 && renderStep1()}
        {currentStep === 2 && renderStep2()}
        {currentStep === 3 && renderStep3()}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '700',
    flex: 1,
    textAlign: 'center',
  },
  stepIndicator: {
    padding: 8,
  },
  stepIndicatorText: {
    fontSize: 14,
    fontWeight: '600',
  },
  progressBar: {
    height: 4,
    marginHorizontal: 20,
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 2,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  stepContainer: {
    flex: 1,
  },
  stepTitle: {
    fontSize: 24,
    fontWeight: '700',
    marginBottom: 8,
  },
  stepDescription: {
    fontSize: 16,
    marginBottom: 24,
    lineHeight: 24,
  },
  constraintsList: {
    gap: 12,
    marginBottom: 32,
  },
  constraintItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    borderRadius: 12,
  },
  constraintInfo: {
    flex: 1,
  },
  constraintName: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  constraintValue: {
    fontSize: 14,
  },
  portfolioContainer: {
    gap: 12,
    marginBottom: 24,
  },
  assetItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    borderRadius: 12,
  },
  assetInfo: {
    flex: 1,
  },
  assetName: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  assetSymbol: {
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 2,
  },
  assetCategory: {
    fontSize: 12,
  },
  assetMetrics: {
    alignItems: 'flex-end',
  },
  assetPercentage: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 4,
  },
  assetReturn: {
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 2,
  },
  assetRisk: {
    fontSize: 12,
  },
  allocationSummary: {
    marginBottom: 24,
  },
  allocationTitle: {
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  allocationWarning: {
    fontSize: 14,
    textAlign: 'center',
    marginTop: 8,
  },
  stressTestContainer: {
    gap: 12,
    marginBottom: 24,
  },
  scenarioItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    borderRadius: 12,
  },
  scenarioName: {
    fontSize: 16,
    fontWeight: '600',
    flex: 1,
  },
  scenarioReturn: {
    fontSize: 18,
    fontWeight: '700',
  },
  portfolioNameContainer: {
    marginBottom: 24,
  },
  portfolioNameLabel: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
  },
  portfolioNameInput: {
    borderWidth: 1,
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
  },
  stepButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  stepBackButton: {
    flex: 1,
    borderWidth: 1,
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
  },
  backButtonText: {
    fontSize: 16,
    fontWeight: '600',
  },
  nextButton: {
    flex: 1,
    borderRadius: 12,
    paddingVertical: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  nextButtonText: {
    fontSize: 16,
    fontWeight: '700',
  },
  saveButton: {
    flex: 1,
    borderRadius: 12,
    paddingVertical: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  saveButtonText: {
    fontSize: 16,
    fontWeight: '700',
  },
});

export default PortfolioBuilderScreen;
