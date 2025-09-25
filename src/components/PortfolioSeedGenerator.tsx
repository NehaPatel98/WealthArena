import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
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

interface PortfolioItem {
  id: string;
  name: string;
  symbol: string;
  percentage: number;
  category: 'stocks' | 'bonds' | 'etf' | 'crypto' | 'reit';
  riskLevel: 'low' | 'medium' | 'high';
  expectedReturn: number;
  description: string;
}

interface PortfolioSeedGeneratorProps {
  userProfile: UserProfile;
  onPortfolioCreate: (portfolio: PortfolioItem[]) => void;
}

const PortfolioSeedGenerator: React.FC<PortfolioSeedGeneratorProps> = ({ 
  userProfile, 
  onPortfolioCreate 
}) => {
  const [generatedPortfolio, setGeneratedPortfolio] = useState<PortfolioItem[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const isDarkMode = true;
  const c = colors[isDarkMode ? 'dark' : 'light'];

  const availableAssets: PortfolioItem[] = useMemo(() => [
    // Conservative/Low Risk
    {
      id: 'spy',
      name: 'SPDR S&P 500 ETF',
      symbol: 'SPY',
      percentage: 0,
      category: 'etf',
      riskLevel: 'low',
      expectedReturn: 8.5,
      description: 'Broad market exposure with low fees'
    },
    {
      id: 'bnd',
      name: 'Vanguard Total Bond Market ETF',
      symbol: 'BND',
      percentage: 0,
      category: 'bonds',
      riskLevel: 'low',
      expectedReturn: 4.2,
      description: 'Diversified bond exposure for stability'
    },
    {
      id: 'vti',
      name: 'Vanguard Total Stock Market ETF',
      symbol: 'VTI',
      percentage: 0,
      category: 'etf',
      riskLevel: 'low',
      expectedReturn: 9.1,
      description: 'Total US stock market exposure'
    },
    // Moderate Risk
    {
      id: 'vxus',
      name: 'Vanguard Total International Stock ETF',
      symbol: 'VXUS',
      percentage: 0,
      category: 'etf',
      riskLevel: 'medium',
      expectedReturn: 7.8,
      description: 'International diversification'
    },
    {
      id: 'vwo',
      name: 'Vanguard Emerging Markets ETF',
      symbol: 'VWO',
      percentage: 0,
      category: 'etf',
      riskLevel: 'medium',
      expectedReturn: 10.2,
      description: 'Emerging markets growth potential'
    },
    {
      id: 'vnq',
      name: 'Vanguard Real Estate ETF',
      symbol: 'VNQ',
      percentage: 0,
      category: 'reit',
      riskLevel: 'medium',
      expectedReturn: 6.5,
      description: 'Real estate investment trust exposure'
    },
    // High Risk/Advanced
    {
      id: 'arkk',
      name: 'ARK Innovation ETF',
      symbol: 'ARKK',
      percentage: 0,
      category: 'etf',
      riskLevel: 'high',
      expectedReturn: 15.3,
      description: 'Innovation and disruptive technology'
    },
    {
      id: 'btc',
      name: 'Bitcoin ETF',
      symbol: 'BTC',
      percentage: 0,
      category: 'crypto',
      riskLevel: 'high',
      expectedReturn: 20.0,
      description: 'Digital currency exposure'
    },
    {
      id: 'qqq',
      name: 'Invesco QQQ Trust',
      symbol: 'QQQ',
      percentage: 0,
      category: 'etf',
      riskLevel: 'high',
      expectedReturn: 12.5,
      description: 'Technology-heavy NASDAQ exposure'
    }
  ], []);

  useEffect(() => {
    generatePortfolio();
  }, [userProfile, generatePortfolio]);

  const generatePortfolio = useCallback(() => {
    setIsGenerating(true);
    
    setTimeout(() => {
      let portfolio: PortfolioItem[] = [];
      
      if (userProfile.experience === 'beginner' || userProfile.riskTolerance === 'conservative') {
        // Conservative portfolio
        portfolio = [
          { ...availableAssets[0], percentage: 60 }, // SPY
          { ...availableAssets[1], percentage: 30 }, // BND
          { ...availableAssets[2], percentage: 10 }, // VTI
        ];
      } else if (userProfile.experience === 'intermediate' || userProfile.riskTolerance === 'moderate') {
        // Moderate portfolio
        portfolio = [
          { ...availableAssets[0], percentage: 40 }, // SPY
          { ...availableAssets[3], percentage: 20 }, // VXUS
          { ...availableAssets[1], percentage: 20 }, // BND
          { ...availableAssets[5], percentage: 10 }, // VNQ
          { ...availableAssets[4], percentage: 10 }, // VWO
        ];
      } else {
        // Advanced portfolio
        portfolio = [
          { ...availableAssets[0], percentage: 30 }, // SPY
          { ...availableAssets[8], percentage: 20 }, // QQQ
          { ...availableAssets[3], percentage: 15 }, // VXUS
          { ...availableAssets[6], percentage: 15 }, // ARKK
          { ...availableAssets[1], percentage: 10 }, // BND
          { ...availableAssets[7], percentage: 10 }, // BTC
        ];
      }
      
      setGeneratedPortfolio(portfolio);
      setIsGenerating(false);
    }, 2000);
  }, [userProfile, availableAssets]);

  const handleCreatePortfolio = () => {
    Alert.alert(
      'Create Portfolio',
      'This will create your personalized portfolio based on your profile. Continue?',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Create', 
          onPress: () => onPortfolioCreate(generatedPortfolio)
        }
      ]
    );
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low': return c.success;
      case 'medium': return c.warning;
      case 'high': return c.danger;
      default: return c.textMuted;
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'stocks': return 'chart-line';
      case 'bonds': return 'shield';
      case 'etf': return 'chart-bar';
      case 'crypto': return 'bitcoin';
      case 'reit': return 'home';
      default: return 'chart-line';
    }
  };

  return (
    <ScrollView style={[styles.container, { backgroundColor: c.background }]}>
      <View style={styles.header}>
        <Text style={[styles.title, { color: c.text }]}>Your Suggested Portfolio</Text>
        <Text style={[styles.subtitle, { color: c.textMuted }]}>
          Generated based on your {userProfile.experience} level and {userProfile.riskTolerance} risk tolerance
        </Text>
      </View>

      {isGenerating ? (
        <View style={styles.generatingContainer}>
          <Icon name="robot" size={48} color={c.primary} />
          <Text style={[styles.generatingText, { color: c.text }]}>
            AI is analyzing your profile...
          </Text>
          <Text style={[styles.generatingSubtext, { color: c.textMuted }]}>
            Creating your personalized portfolio
          </Text>
        </View>
      ) : (
        <>
          <View style={styles.portfolioContainer}>
            {generatedPortfolio.map((item, index) => (
              <View key={item.id} style={[styles.portfolioItem, { backgroundColor: c.surface }]}>
                <View style={styles.itemHeader}>
                  <View style={styles.itemInfo}>
                    <View style={[styles.itemIcon, { backgroundColor: c.primary }]}>
                      <Icon name={getCategoryIcon(item.category)} size={20} color={c.background} />
                    </View>
                    <View style={styles.itemDetails}>
                      <Text style={[styles.itemName, { color: c.text }]}>{item.name}</Text>
                      <Text style={[styles.itemSymbol, { color: c.textMuted }]}>{item.symbol}</Text>
                    </View>
                  </View>
                  <View style={styles.itemStats}>
                    <Text style={[styles.itemPercentage, { color: c.primary }]}>
                      {item.percentage}%
                    </Text>
                    <View style={[styles.riskBadge, { backgroundColor: getRiskColor(item.riskLevel) + '20' }]}>
                      <Text style={[styles.riskText, { color: getRiskColor(item.riskLevel) }]}>
                        {item.riskLevel.toUpperCase()}
                      </Text>
                    </View>
                  </View>
                </View>
                
                <View style={styles.itemDescription}>
                  <Text style={[styles.descriptionText, { color: c.textMuted }]}>
                    {item.description}
                  </Text>
                </View>
                
                <View style={styles.itemMetrics}>
                  <View style={styles.metric}>
                    <Text style={[styles.metricLabel, { color: c.textMuted }]}>Expected Return</Text>
                    <Text style={[styles.metricValue, { color: c.success }]}>
                      {item.expectedReturn}%
                    </Text>
                  </View>
                  <View style={styles.metric}>
                    <Text style={[styles.metricLabel, { color: c.textMuted }]}>Allocation</Text>
                    <Text style={[styles.metricValue, { color: c.text }]}>
                      {item.percentage}%
                    </Text>
                  </View>
                </View>
              </View>
            ))}
          </View>

          <View style={styles.summaryCard}>
            <Text style={[styles.summaryTitle, { color: c.text }]}>Portfolio Summary</Text>
            <View style={styles.summaryStats}>
              <View style={styles.summaryItem}>
                <Text style={[styles.summaryLabel, { color: c.textMuted }]}>Total Allocation</Text>
                <Text style={[styles.summaryValue, { color: c.text }]}>100%</Text>
              </View>
              <View style={styles.summaryItem}>
                <Text style={[styles.summaryLabel, { color: c.textMuted }]}>Risk Level</Text>
                <Text style={[styles.summaryValue, { color: getRiskColor(userProfile.riskTolerance) }]}>
                  {userProfile.riskTolerance.toUpperCase()}
                </Text>
              </View>
              <View style={styles.summaryItem}>
                <Text style={[styles.summaryLabel, { color: c.textMuted }]}>Expected Return</Text>
                <Text style={[styles.summaryValue, { color: c.success }]}>
                  {generatedPortfolio.reduce((acc, item) => acc + (item.expectedReturn * item.percentage / 100), 0).toFixed(1)}%
                </Text>
              </View>
            </View>
          </View>

          <TouchableOpacity
            style={[styles.createButton, { backgroundColor: c.primary }]}
            onPress={handleCreatePortfolio}
          >
            <Icon name="plus" size={20} color={c.background} />
            <Text style={[styles.createButtonText, { color: c.background }]}>
              Create This Portfolio
            </Text>
          </TouchableOpacity>
        </>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    padding: 20,
    paddingBottom: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    lineHeight: 22,
  },
  generatingContainer: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  generatingText: {
    fontSize: 18,
    fontWeight: '600',
    marginTop: 16,
    marginBottom: 8,
  },
  generatingSubtext: {
    fontSize: 14,
  },
  portfolioContainer: {
    paddingHorizontal: 20,
    gap: 16,
  },
  portfolioItem: {
    borderRadius: 12,
    padding: 16,
  },
  itemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  itemInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  itemIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  itemDetails: {
    flex: 1,
  },
  itemName: {
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 2,
  },
  itemSymbol: {
    fontSize: 14,
  },
  itemStats: {
    alignItems: 'flex-end',
  },
  itemPercentage: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 4,
  },
  riskBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  riskText: {
    fontSize: 10,
    fontWeight: '700',
  },
  itemDescription: {
    marginBottom: 12,
  },
  descriptionText: {
    fontSize: 14,
    lineHeight: 20,
  },
  itemMetrics: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  metric: {
    alignItems: 'center',
  },
  metricLabel: {
    fontSize: 12,
    marginBottom: 4,
  },
  metricValue: {
    fontSize: 16,
    fontWeight: '700',
  },
  summaryCard: {
    margin: 20,
    padding: 20,
    borderRadius: 12,
    backgroundColor: '#1a1a1a',
  },
  summaryTitle: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 16,
  },
  summaryStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  summaryItem: {
    alignItems: 'center',
  },
  summaryLabel: {
    fontSize: 12,
    marginBottom: 4,
  },
  summaryValue: {
    fontSize: 16,
    fontWeight: '700',
  },
  createButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    margin: 20,
    paddingVertical: 16,
    borderRadius: 12,
    gap: 8,
  },
  createButtonText: {
    fontSize: 16,
    fontWeight: '700',
  },
});

export default PortfolioSeedGenerator;
