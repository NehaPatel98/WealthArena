/**
 * Game Play Screen
 * Beginner-friendly trading game with educational tooltips
 */

import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, Pressable, Alert, Modal } from 'react-native';
import { useRouter, Stack, useLocalSearchParams } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { 
  useTheme, 
  Text, 
  Card, 
  Button, 
  Badge, 
  FAB, 
  tokens 
} from '@/src/design-system';
import CandlestickChart from '../components/CandlestickChart';

// Tutorial steps for beginner mode
const TUTORIAL_STEPS = [
  {
    id: 'welcome',
    title: 'Welcome to Trading!',
    description: 'Let\'s learn the basics of trading step by step. This game uses real historical market data so you can practice risk-free.',
    icon: 'hand-right',
  },
  {
    id: 'reading-charts',
    title: 'Reading Price Charts',
    description: 'The green line shows the price going UP. The red line shows the price going DOWN. Each point represents the price at a specific time.',
    icon: 'analytics',
  },
  {
    id: 'buying',
    title: 'How to Buy',
    description: 'When you think the price will go UP, you BUY. You\'ll make money if the price increases after you buy.',
    icon: 'trending-up',
  },
  {
    id: 'selling',
    title: 'How to Sell',
    description: 'When you think the price will go DOWN (or you want to lock in profits), you SELL. Selling closes your position.',
    icon: 'trending-down',
  },
  {
    id: 'portfolio',
    title: 'Your Portfolio',
    description: 'Your portfolio shows all the investments you own and how much money you have. The P&L (Profit & Loss) shows if you\'re winning or losing.',
    icon: 'wallet',
  },
];

export default function GamePlayScreen() {
  const router = useRouter();
  const params = useLocalSearchParams();
  const { theme } = useTheme();
  
  // Parse params
  const symbols = (params.symbols as string)?.split(',') || ['SPY'];
  const mode = (params.mode as string) || 'beginner';
  const difficulty = (params.difficulty as string) || 'easy';
  const portfolioName = (params.portfolioName as string) || 'My Portfolio';
  
  const [currentSymbol, setCurrentSymbol] = useState(symbols[0]);
  const [balance, setBalance] = useState(difficulty === 'easy' ? 100000 : difficulty === 'medium' ? 50000 : 25000);
  const [positions, setPositions] = useState<any[]>([]);
  const [showTutorial, setShowTutorial] = useState(mode === 'beginner');
  const [tutorialStep, setTutorialStep] = useState(0);
  const [showHelp, setShowHelp] = useState(false);
  const [currentPrice, setCurrentPrice] = useState(450.00);
  const [priceHistory, setPriceHistory] = useState<number[]>([450.00]);
  const [pnl, setPnl] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  
  // Generate chart data from price history
  const chartData = priceHistory.slice(-30).map((price, index) => ({
    time: `T${index + 1}`,
    open: index > 0 ? priceHistory[priceHistory.length - 30 + index - 1] : price,
    high: price + Math.random() * 2,
    low: price - Math.random() * 2,
    close: price
  }));

  // Simulate price movements
  useEffect(() => {
    if (!isPlaying) return;
    
    const interval = setInterval(() => {
      const change = (Math.random() - 0.5) * 5; // Random price movement
      const newPrice = Math.max(currentPrice + change, 1);
      setCurrentPrice(newPrice);
      setPriceHistory(prev => [...prev.slice(-50), newPrice]); // Keep last 50 points
      
      // Update positions P&L
      const totalPnL = positions.reduce((sum, pos) => {
        return sum + ((newPrice - pos.entryPrice) * pos.quantity);
      }, 0);
      setPnl(totalPnL);
    }, mode === 'beginner' ? 2000 : mode === 'standard' ? 1000 : 500);
    
    return () => clearInterval(interval);
  }, [isPlaying, currentPrice, positions, mode]);

  const handleBuy = () => {
    if (mode === 'beginner' && tutorialStep < TUTORIAL_STEPS.length && !showTutorial) {
      Alert.alert(
        '💡 Trading Tip',
        'You\'re buying at $' + currentPrice.toFixed(2) + '. You\'ll make money if the price goes UP from here!',
        [{ text: 'Got it!', onPress: () => executeBuy() }]
      );
    } else {
      executeBuy();
    }
  };

  const executeBuy = () => {
    const quantity = 10; // Fixed quantity for simplicity
    const cost = currentPrice * quantity;
    
    if (balance < cost) {
      Alert.alert('Insufficient Funds', 'You don\'t have enough cash to buy this position.');
      return;
    }
    
    setBalance(prev => prev - cost);
    setPositions(prev => [...prev, {
      symbol: currentSymbol,
      quantity: quantity,
      entryPrice: currentPrice,
      timestamp: new Date().toISOString()
    }]);
    
    if (mode === 'beginner') {
      Alert.alert(
        '✅ Purchase Complete!',
        `You bought ${quantity} shares of ${currentSymbol} at $${currentPrice.toFixed(2)}\n\nNow wait for the price to go UP to make profit!`
      );
    }
  };

  const handleSell = () => {
    if (positions.length === 0) {
      Alert.alert('No Positions', 'You don\'t have any positions to sell.');
      return;
    }
    
    const position = positions[0];
    const profit = (currentPrice - position.entryPrice) * position.quantity;
    const proceeds = currentPrice * position.quantity;
    
    setBalance(prev => prev + proceeds);
    setPositions(prev => prev.slice(1));
    setPnl(prev => prev - profit);
    
    if (mode === 'beginner') {
      const profitText = profit >= 0 
        ? `You made $${profit.toFixed(2)} profit! 🎉` 
        : `You lost $${Math.abs(profit).toFixed(2)} 😔`;
      
      Alert.alert(
        '✅ Position Closed!',
        `You sold ${position.quantity} shares of ${position.symbol}\n\n${profitText}`
      );
    }
  };

  const nextTutorialStep = () => {
    if (tutorialStep < TUTORIAL_STEPS.length - 1) {
      setTutorialStep(prev => prev + 1);
    } else {
      setShowTutorial(false);
      setIsPlaying(true);
    }
  };

  const skipTutorial = () => {
    setShowTutorial(false);
    setIsPlaying(true);
  };

  const currentTutorial = TUTORIAL_STEPS[tutorialStep];

  const handleExit = () => {
    if (isPlaying) {
      Alert.alert(
        'Exit Game?',
        'You are currently in a trading session. Are you sure you want to exit? Your progress will be lost.',
        [
          { text: 'Cancel', style: 'cancel' },
          { 
            text: 'Exit', 
            style: 'destructive',
            onPress: () => {
              setIsPlaying(false);
              setShowTutorial(false);
              setTutorialStep(0);
              router.back();
            }
          }
        ]
      );
    } else {
      router.back();
    }
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.bg }]} edges={['top']}>
      <Stack.Screen options={{ headerShown: false }} />
      
      {/* Custom Header */}
      <View style={[styles.header, { backgroundColor: theme.bg, borderBottomColor: theme.border }]}>
        <Pressable onPress={handleExit} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color={theme.text} />
        </Pressable>
        <Text variant="h3" weight="semibold" style={styles.headerTitle}>{portfolioName}</Text>
        <View style={styles.headerRight}>
          <Pressable onPress={() => setShowHelp(true)} style={styles.helpButton}>
            <Ionicons name="help-circle-outline" size={24} color={theme.primary} />
          </Pressable>
          {isPlaying && (
            <Pressable onPress={handleExit} style={styles.exitButton}>
              <Ionicons name="stop-circle-outline" size={24} color={theme.danger} />
            </Pressable>
          )}
        </View>
      </View>
      
      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        {/* Balance Card */}
        <Card style={styles.balanceCard} elevation="med">
          <View style={styles.balanceRow}>
            <View style={{ flex: 1 }}>
              <Text variant="small" muted>
                {mode === 'beginner' ? 'Cash Available' : 'Balance'}
              </Text>
              <Text variant="h2" weight="bold">${balance.toFixed(2)}</Text>
            </View>
            <View style={{ flex: 1 }}>
              <Text variant="small" muted>
                {mode === 'beginner' ? 'Profit/Loss' : 'P&L'}
              </Text>
              <Text 
                variant="h3" 
                weight="bold"
                color={pnl >= 0 ? theme.success : theme.danger}
              >
                {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
              </Text>
            </View>
            <View style={{ flex: 1 }}>
              <Text variant="small" muted>
                {mode === 'beginner' ? 'Open Trades' : 'Positions'}
              </Text>
              <Text variant="h3" weight="bold">{positions.length}</Text>
            </View>
          </View>
        </Card>

        {/* Symbol Selector */}
        {symbols.length > 1 && (
          <Card style={styles.symbolSelector}>
            <Text variant="small" muted>Trading</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              <View style={styles.symbolButtons}>
                {symbols.map(symbol => (
                  <Pressable
                    key={symbol}
                    onPress={() => setCurrentSymbol(symbol)}
                  >
                    <Badge
                      variant={currentSymbol === symbol ? 'primary' : 'secondary'}
                      size="medium"
                    >
                      {symbol}
                    </Badge>
                  </Pressable>
                ))}
              </View>
            </ScrollView>
          </Card>
        )}

        {/* Price Chart - Simplified for beginners */}
        <Card style={styles.chartCard}>
          <View style={styles.chartHeader}>
            <View>
              <Text variant="h3" weight="bold">{currentSymbol}</Text>
              <Text variant="small" muted>Current Price</Text>
            </View>
            <View style={{ alignItems: 'flex-end' }}>
              <Text variant="h2" weight="bold" color={theme.primary}>
                ${currentPrice.toFixed(2)}
              </Text>
              {priceHistory.length > 1 && (
                <Badge 
                  variant={currentPrice >= priceHistory[priceHistory.length - 2] ? 'success' : 'danger'}
                  size="small"
                >
                  {currentPrice >= priceHistory[priceHistory.length - 2] ? '↑' : '↓'}
                  {Math.abs(((currentPrice - priceHistory[priceHistory.length - 2]) / priceHistory[priceHistory.length - 2]) * 100).toFixed(2)}%
                </Badge>
              )}
            </View>
          </View>

          {/* Price Chart */}
          {mode === 'beginner' && (
            <Text variant="xs" muted center style={styles.chartLabel}>
              📈 Green = Price Going UP | 📉 Red = Price Going DOWN
            </Text>
          )}
          
          {chartData.length > 0 && (
            <CandlestickChart
              data={chartData}
              chartType="daily"
              beginnerMode={mode === 'beginner'}
              showTooltip={true}
            />
          )}
        </Card>

        {/* Beginner Tips */}
        {mode === 'beginner' && !showTutorial && (
          <Card style={StyleSheet.flatten([styles.tipCard, { backgroundColor: theme.primary + '10' }])}>
            <Ionicons name="bulb" size={24} color={theme.primary} />
            <View style={{ flex: 1 }}>
              <Text variant="body" weight="semibold" color={theme.primary}>Quick Tip</Text>
              <Text variant="small" muted>
                {positions.length === 0 
                  ? 'Buy when you think the price will go UP. You can always sell later!'
                  : 'You have open positions. Sell when you want to lock in profits or cut losses.'}
              </Text>
            </View>
          </Card>
        )}

        {/* Trading Actions */}
        <Card style={styles.actionsCard}>
          <Text variant="h3" weight="semibold">
            {mode === 'beginner' ? 'What would you like to do?' : 'Trading Actions'}
          </Text>
          {mode === 'beginner' && (
            <Text variant="xs" muted>
              Buy when you think the price will go up. Sell to close and lock in profits/losses.
            </Text>
          )}
          
          <View style={styles.actionButtons}>
            <Button
              variant="primary"
              size="large"
              onPress={handleBuy}
              disabled={!isPlaying}
              icon={<Ionicons name="arrow-up" size={24} color="#FFFFFF" />}
              style={{ flex: 1 }}
            >
              {mode === 'beginner' ? 'Enter Trade' : 'Buy'}
            </Button>
            
            <Button
              variant="danger"
              size="large"
              onPress={handleSell}
              disabled={!isPlaying || positions.length === 0}
              icon={<Ionicons name="arrow-down" size={24} color="#FFFFFF" />}
              style={{ flex: 1 }}
            >
              {mode === 'beginner' ? 'Exit Trade' : 'Sell'}
            </Button>
          </View>

          {!isPlaying && !showTutorial && (
            <Button
              variant="primary"
              size="medium"
              onPress={() => setIsPlaying(true)}
              icon={<Ionicons name="play" size={20} color={theme.bg} />}
              fullWidth
            >
              Start Game
            </Button>
          )}

          {/* End Game Button */}
          {isPlaying && (
            <Pressable 
              onPress={handleExit}
              style={[styles.endGameButton, { backgroundColor: theme.danger + '20', borderColor: theme.danger }]}
            >
              <Ionicons name="stop-circle" size={20} color={theme.danger} />
              <Text variant="body" weight="semibold" color={theme.danger}>
                End Game
              </Text>
            </Pressable>
          )}
        </Card>

        {/* Positions List */}
        {positions.length > 0 && (
          <Card style={styles.positionsCard}>
            <Text variant="h3" weight="semibold">Your Positions</Text>
            
            {positions.map((position, index) => {
              const positionPnL = (currentPrice - position.entryPrice) * position.quantity;
              return (
                <Card key={index} style={styles.positionItem} elevation="low">
                  <View style={styles.positionHeader}>
                    <Badge variant="primary" size="medium">{position.symbol}</Badge>
                    <Badge 
                      variant={positionPnL >= 0 ? 'success' : 'danger'} 
                      size="small"
                    >
                      {positionPnL >= 0 ? '+' : ''}${positionPnL.toFixed(2)}
                    </Badge>
                  </View>
                  <View style={styles.positionDetails}>
                    <Text variant="small" muted>Qty: {position.quantity}</Text>
                    <Text variant="small" muted>Entry: ${position.entryPrice.toFixed(2)}</Text>
                    <Text variant="small" muted>Current: ${currentPrice.toFixed(2)}</Text>
                  </View>
                </Card>
              );
            })}
          </Card>
        )}

        <View style={{ height: tokens.spacing.xl }} />
      </ScrollView>

      {/* Tutorial Modal */}
      <Modal
        visible={showTutorial}
        transparent
        animationType="slide"
      >
        <View style={styles.modalOverlay}>
          <SafeAreaView style={{ flex: 1, justifyContent: 'flex-end' }}>
            <Card style={styles.tutorialModal} elevation="high">
              <View style={[styles.tutorialIcon, { backgroundColor: theme.primary + '20' }]}>
                <Ionicons name={currentTutorial.icon as any} size={48} color={theme.primary} />
              </View>
              
              <Text variant="h2" weight="bold" center>
                {currentTutorial.title}
              </Text>
              
              <Text variant="body" center muted>
                {currentTutorial.description}
              </Text>
              
              <View style={styles.tutorialProgress}>
                {TUTORIAL_STEPS.map((_, index) => (
                  <View 
                    key={index}
                    style={[
                      styles.progressDot,
                      { 
                        backgroundColor: index === tutorialStep ? theme.primary : theme.border 
                      }
                    ]} 
                  />
                ))}
              </View>
              
              <View style={styles.tutorialActions}>
                <Button
                  variant="ghost"
                  size="medium"
                  onPress={skipTutorial}
                >
                  Skip Tutorial
                </Button>
                
                <Button
                  variant="primary"
                  size="medium"
                  onPress={nextTutorialStep}
                  icon={<Ionicons name={tutorialStep < TUTORIAL_STEPS.length - 1 ? "arrow-forward" : "checkmark"} size={20} color={theme.bg} />}
                >
                  {tutorialStep < TUTORIAL_STEPS.length - 1 ? 'Next' : 'Start Trading!'}
                </Button>
              </View>
            </Card>
          </SafeAreaView>
        </View>
      </Modal>

      {/* Help Modal */}
      <Modal
        visible={showHelp}
        transparent
        animationType="fade"
        onRequestClose={() => setShowHelp(false)}
      >
        <Pressable style={styles.modalOverlay} onPress={() => setShowHelp(false)}>
          <SafeAreaView style={{ flex: 1, justifyContent: 'center', padding: tokens.spacing.md }}>
            <Pressable onPress={(e) => e.stopPropagation()}>
              <Card style={styles.helpModal} elevation="high">
                <Text variant="h2" weight="bold">Help & Tips</Text>
                
                <View style={styles.helpItem}>
                  <Ionicons name="trending-up" size={24} color={theme.success} />
                  <View style={{ flex: 1 }}>
                    <Text variant="body" weight="semibold">Buying</Text>
                    <Text variant="small" muted>
                      Buy when you think the price will increase. You profit when you sell at a higher price.
                    </Text>
                  </View>
                </View>
                
                <View style={styles.helpItem}>
                  <Ionicons name="trending-down" size={24} color={theme.danger} />
                  <View style={{ flex: 1 }}>
                    <Text variant="body" weight="semibold">Selling</Text>
                    <Text variant="small" muted>
                      Sell to close your position and lock in profits or cut losses.
                    </Text>
                  </View>
                </View>
                
                <View style={styles.helpItem}>
                  <Ionicons name="wallet" size={24} color={theme.primary} />
                  <View style={{ flex: 1 }}>
                    <Text variant="body" weight="semibold">P&L (Profit & Loss)</Text>
                    <Text variant="small" muted>
                      Shows how much you're winning or losing on your current positions.
                    </Text>
                  </View>
                </View>
                
                <Button
                  variant="primary"
                  size="medium"
                  onPress={() => setShowHelp(false)}
                  fullWidth
                >
                  Got it!
                </Button>
              </Card>
            </Pressable>
          </SafeAreaView>
        </Pressable>
      </Modal>
      
      <FAB onPress={() => router.push('/ai-chat')} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: tokens.spacing.md,
    paddingVertical: tokens.spacing.sm,
    borderBottomWidth: 1,
    height: 56,
  },
  backButton: {
    padding: tokens.spacing.xs,
    marginLeft: -tokens.spacing.xs,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: tokens.spacing.sm,
  },
  helpButton: {
    padding: tokens.spacing.xs,
  },
  exitButton: {
    padding: tokens.spacing.xs,
  },
  endGameButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: tokens.spacing.xs,
    padding: tokens.spacing.sm,
    marginTop: tokens.spacing.sm,
    borderRadius: tokens.radius.md,
    borderWidth: 1,
  },
  headerTitle: {
    flex: 1,
    textAlign: 'center',
    marginHorizontal: tokens.spacing.md,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: tokens.spacing.md,
    gap: tokens.spacing.md,
  },
  balanceCard: {
    gap: tokens.spacing.sm,
  },
  balanceRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  symbolSelector: {
    gap: tokens.spacing.xs,
  },
  symbolButtons: {
    flexDirection: 'row',
    gap: tokens.spacing.xs,
  },
  chartCard: {
    gap: tokens.spacing.md,
  },
  chartHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  chartContainer: {
    borderRadius: tokens.radius.md,
    padding: tokens.spacing.sm,
    gap: tokens.spacing.xs,
  },
  chartLabel: {
    marginBottom: tokens.spacing.xs,
  },
  chartArea: {
    height: 150,
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 2,
  },
  chartBar: {
    flex: 1,
    borderRadius: 2,
  },
  tipCard: {
    flexDirection: 'row',
    gap: tokens.spacing.md,
    alignItems: 'flex-start',
  },
  actionsCard: {
    gap: tokens.spacing.md,
  },
  actionButtons: {
    flexDirection: 'row',
    gap: tokens.spacing.md,
  },
  positionsCard: {
    gap: tokens.spacing.md,
  },
  positionItem: {
    gap: tokens.spacing.sm,
  },
  positionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  positionDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  tutorialModal: {
    margin: tokens.spacing.md,
    padding: tokens.spacing.lg,
    gap: tokens.spacing.md,
    alignItems: 'center',
  },
  tutorialIcon: {
    width: 100,
    height: 100,
    borderRadius: 50,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tutorialProgress: {
    flexDirection: 'row',
    gap: tokens.spacing.xs,
  },
  progressDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  tutorialActions: {
    flexDirection: 'row',
    gap: tokens.spacing.md,
    width: '100%',
  },
  helpModal: {
    gap: tokens.spacing.md,
  },
  helpItem: {
    flexDirection: 'row',
    gap: tokens.spacing.md,
    alignItems: 'flex-start',
  },
});

