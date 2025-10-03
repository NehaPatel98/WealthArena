import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
} from 'react-native';
import { Stack } from 'expo-router';
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Clock,
  BarChart3,
  Bot,
  User,
  ArrowUpRight,
  ArrowDownRight,
} from 'lucide-react-native';
import Colors from '@/constants/colors';

type OrderType = 'market' | 'limit';
type OrderSide = 'buy' | 'sell';

interface Trade {
  id: string;
  symbol: string;
  side: OrderSide;
  type: OrderType;
  quantity: number;
  price: number;
  timestamp: Date;
  cost: number;
}

const STOCKS = [
  { symbol: 'AAPL', name: 'Apple Inc.', price: 178.45, change: 2.34 },
  { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 142.67, change: -1.23 },
  { symbol: 'MSFT', name: 'Microsoft Corp.', price: 412.89, change: 5.67 },
  { symbol: 'TSLA', name: 'Tesla Inc.', price: 242.15, change: -3.45 },
];

export default function TradeSimulatorScreen() {
  const [selectedStock, setSelectedStock] = useState(STOCKS[0]);
  const [orderType, setOrderType] = useState<OrderType>('market');
  const [orderSide, setOrderSide] = useState<OrderSide>('buy');
  const [quantity, setQuantity] = useState('10');
  const [limitPrice, setLimitPrice] = useState('');
  const [trades, setTrades] = useState<Trade[]>([]);

  const handlePlaceOrder = () => {
    const newTrade: Trade = {
      id: Date.now().toString(),
      symbol: selectedStock.symbol,
      side: orderSide,
      type: orderType,
      quantity: parseInt(quantity) || 0,
      price: orderType === 'market' ? selectedStock.price : parseFloat(limitPrice) || 0,
      timestamp: new Date(),
      cost: 2.5,
    };
    setTrades([newTrade, ...trades]);
    console.log('Order placed:', newTrade);
  };

  const totalValue = parseInt(quantity) * (orderType === 'market' ? selectedStock.price : parseFloat(limitPrice) || 0);
  const estimatedCost = 2.5;

  return (
    <View style={styles.container}>
      <Stack.Screen
        options={{
          title: 'Trade Simulator',
          headerStyle: { backgroundColor: Colors.background },
          headerTintColor: Colors.text,
        }}
      />
      <ScrollView showsVerticalScrollIndicator={false}>
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Select Stock</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.stocksScroll}>
            {STOCKS.map((stock) => (
              <TouchableOpacity
                key={stock.symbol}
                style={[
                  styles.stockCard,
                  selectedStock.symbol === stock.symbol && styles.stockCardSelected,
                ]}
                onPress={() => setSelectedStock(stock)}
              >
                <Text style={styles.stockSymbol}>{stock.symbol}</Text>
                <Text style={styles.stockPrice}>${stock.price.toFixed(2)}</Text>
                <View style={styles.stockChange}>
                  {stock.change >= 0 ? (
                    <ArrowUpRight size={14} color={Colors.chartGreen} />
                  ) : (
                    <ArrowDownRight size={14} color={Colors.chartRed} />
                  )}
                  <Text
                    style={[
                      styles.stockChangeText,
                      { color: stock.change >= 0 ? Colors.chartGreen : Colors.chartRed },
                    ]}
                  >
                    {Math.abs(stock.change).toFixed(2)}%
                  </Text>
                </View>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Order Type</Text>
          <View style={styles.buttonGroup}>
            <TouchableOpacity
              style={[styles.typeButton, orderType === 'market' && styles.typeButtonActive]}
              onPress={() => setOrderType('market')}
            >
              <Text
                style={[styles.typeButtonText, orderType === 'market' && styles.typeButtonTextActive]}
              >
                Market
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.typeButton, orderType === 'limit' && styles.typeButtonActive]}
              onPress={() => setOrderType('limit')}
            >
              <Text
                style={[styles.typeButtonText, orderType === 'limit' && styles.typeButtonTextActive]}
              >
                Limit
              </Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Side</Text>
          <View style={styles.buttonGroup}>
            <TouchableOpacity
              style={[styles.sideButton, styles.buyButton, orderSide === 'buy' && styles.buyButtonActive]}
              onPress={() => setOrderSide('buy')}
            >
              <TrendingUp size={20} color={orderSide === 'buy' ? Colors.text : Colors.chartGreen} />
              <Text
                style={[styles.sideButtonText, orderSide === 'buy' && styles.sideButtonTextActive]}
              >
                Buy
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.sideButton, styles.sellButton, orderSide === 'sell' && styles.sellButtonActive]}
              onPress={() => setOrderSide('sell')}
            >
              <TrendingDown size={20} color={orderSide === 'sell' ? Colors.text : Colors.chartRed} />
              <Text
                style={[styles.sideButtonText, orderSide === 'sell' && styles.sideButtonTextActive]}
              >
                Sell
              </Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Quantity</Text>
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.input}
              value={quantity}
              onChangeText={setQuantity}
              keyboardType="numeric"
              placeholder="Enter quantity"
              placeholderTextColor={Colors.textMuted}
            />
            <Text style={styles.inputLabel}>shares</Text>
          </View>
        </View>

        {orderType === 'limit' && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Limit Price</Text>
            <View style={styles.inputContainer}>
              <DollarSign size={20} color={Colors.textMuted} />
              <TextInput
                style={styles.input}
                value={limitPrice}
                onChangeText={setLimitPrice}
                keyboardType="decimal-pad"
                placeholder="Enter limit price"
                placeholderTextColor={Colors.textMuted}
              />
            </View>
          </View>
        )}

        <View style={styles.summaryCard}>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Order Value</Text>
            <Text style={styles.summaryValue}>${totalValue.toFixed(2)}</Text>
          </View>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Transaction Cost</Text>
            <Text style={styles.summaryValue}>${estimatedCost.toFixed(2)}</Text>
          </View>
          <View style={styles.summaryDivider} />
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabelBold}>Total</Text>
            <Text style={styles.summaryValueBold}>${(totalValue + estimatedCost).toFixed(2)}</Text>
          </View>
        </View>

        <TouchableOpacity
          style={[styles.placeOrderButton, orderSide === 'buy' ? styles.buyOrderButton : styles.sellOrderButton]}
          onPress={handlePlaceOrder}
        >
          <Text style={styles.placeOrderText}>
            Place {orderSide === 'buy' ? 'Buy' : 'Sell'} Order
          </Text>
        </TouchableOpacity>

        <View style={styles.section}>
          <View style={styles.comparisonHeader}>
            <Text style={styles.sectionTitle}>Performance Comparison</Text>
            <View style={styles.comparisonLegend}>
              <View style={styles.legendItem}>
                <User size={16} color={Colors.secondary} />
                <Text style={styles.legendText}>You</Text>
              </View>
              <View style={styles.legendItem}>
                <Bot size={16} color={Colors.gold} />
                <Text style={styles.legendText}>Agent</Text>
              </View>
            </View>
          </View>
          <View style={styles.comparisonCard}>
            <View style={styles.comparisonRow}>
              <Text style={styles.comparisonLabel}>Total Return</Text>
              <View style={styles.comparisonValues}>
                <Text style={[styles.comparisonValue, { color: Colors.secondary }]}>+12.4%</Text>
                <Text style={[styles.comparisonValue, { color: Colors.gold }]}>+15.8%</Text>
              </View>
            </View>
            <View style={styles.comparisonRow}>
              <Text style={styles.comparisonLabel}>Win Rate</Text>
              <View style={styles.comparisonValues}>
                <Text style={[styles.comparisonValue, { color: Colors.secondary }]}>68%</Text>
                <Text style={[styles.comparisonValue, { color: Colors.gold }]}>72%</Text>
              </View>
            </View>
            <View style={styles.comparisonRow}>
              <Text style={styles.comparisonLabel}>Avg Trade</Text>
              <View style={styles.comparisonValues}>
                <Text style={[styles.comparisonValue, { color: Colors.secondary }]}>$245</Text>
                <Text style={[styles.comparisonValue, { color: Colors.gold }]}>$312</Text>
              </View>
            </View>
          </View>
        </View>

        {trades.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Recent Trades</Text>
            {trades.slice(0, 5).map((trade) => (
              <View key={trade.id} style={styles.tradeCard}>
                <View style={styles.tradeHeader}>
                  <View style={styles.tradeLeft}>
                    <Text style={styles.tradeSymbol}>{trade.symbol}</Text>
                    <View
                      style={[
                        styles.tradeSideBadge,
                        { backgroundColor: trade.side === 'buy' ? Colors.chartGreen + '20' : Colors.chartRed + '20' },
                      ]}
                    >
                      <Text
                        style={[
                          styles.tradeSideText,
                          { color: trade.side === 'buy' ? Colors.chartGreen : Colors.chartRed },
                        ]}
                      >
                        {trade.side.toUpperCase()}
                      </Text>
                    </View>
                  </View>
                  <Text style={styles.tradeValue}>${(trade.quantity * trade.price).toFixed(2)}</Text>
                </View>
                <View style={styles.tradeDetails}>
                  <View style={styles.tradeDetailItem}>
                    <Clock size={14} color={Colors.textMuted} />
                    <Text style={styles.tradeDetailText}>
                      {trade.timestamp.toLocaleTimeString()}
                    </Text>
                  </View>
                  <Text style={styles.tradeDetailText}>
                    {trade.quantity} @ ${trade.price.toFixed(2)}
                  </Text>
                </View>
              </View>
            ))}
          </View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  section: {
    padding: 24,
    paddingBottom: 0,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: Colors.text,
    marginBottom: 16,
  },
  stocksScroll: {
    marginBottom: 24,
  },
  stockCard: {
    backgroundColor: Colors.surface,
    padding: 16,
    borderRadius: 16,
    marginRight: 12,
    minWidth: 120,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  stockCardSelected: {
    borderColor: Colors.secondary,
  },
  stockSymbol: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.text,
    marginBottom: 8,
  },
  stockPrice: {
    fontSize: 20,
    fontWeight: '600' as const,
    color: Colors.text,
    marginBottom: 4,
  },
  stockChange: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  stockChangeText: {
    fontSize: 12,
    fontWeight: '600' as const,
  },
  buttonGroup: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 24,
  },
  typeButton: {
    flex: 1,
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 12,
    backgroundColor: Colors.surface,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  typeButtonActive: {
    borderColor: Colors.secondary,
    backgroundColor: Colors.secondary + '20',
  },
  typeButtonText: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.textSecondary,
  },
  typeButtonTextActive: {
    color: Colors.secondary,
  },
  sideButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 16,
    borderRadius: 12,
    borderWidth: 2,
  },
  buyButton: {
    backgroundColor: Colors.surface,
    borderColor: Colors.chartGreen + '40',
  },
  buyButtonActive: {
    backgroundColor: Colors.chartGreen,
    borderColor: Colors.chartGreen,
  },
  sellButton: {
    backgroundColor: Colors.surface,
    borderColor: Colors.chartRed + '40',
  },
  sellButtonActive: {
    backgroundColor: Colors.chartRed,
    borderColor: Colors.chartRed,
  },
  sideButtonText: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.textSecondary,
  },
  sideButtonTextActive: {
    color: Colors.text,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.surface,
    borderRadius: 12,
    paddingHorizontal: 16,
    height: 56,
    marginBottom: 24,
    gap: 8,
  },
  input: {
    flex: 1,
    fontSize: 16,
    color: Colors.text,
  },
  inputLabel: {
    fontSize: 14,
    color: Colors.textMuted,
  },
  summaryCard: {
    backgroundColor: Colors.surface,
    marginHorizontal: 24,
    padding: 20,
    borderRadius: 16,
    marginBottom: 24,
    gap: 12,
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
  summaryLabelBold: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  summaryValueBold: {
    fontSize: 20,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  summaryDivider: {
    height: 1,
    backgroundColor: Colors.border,
    marginVertical: 4,
  },
  placeOrderButton: {
    marginHorizontal: 24,
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 24,
  },
  buyOrderButton: {
    backgroundColor: Colors.chartGreen,
  },
  sellOrderButton: {
    backgroundColor: Colors.chartRed,
  },
  placeOrderText: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  comparisonHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  comparisonLegend: {
    flexDirection: 'row',
    gap: 16,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  legendText: {
    fontSize: 12,
    color: Colors.textSecondary,
  },
  comparisonCard: {
    backgroundColor: Colors.surface,
    padding: 20,
    borderRadius: 16,
    gap: 16,
    marginBottom: 24,
  },
  comparisonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  comparisonLabel: {
    fontSize: 14,
    color: Colors.textSecondary,
  },
  comparisonValues: {
    flexDirection: 'row',
    gap: 24,
  },
  comparisonValue: {
    fontSize: 16,
    fontWeight: '600' as const,
    minWidth: 60,
    textAlign: 'right',
  },
  tradeCard: {
    backgroundColor: Colors.surface,
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    gap: 12,
  },
  tradeHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  tradeLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  tradeSymbol: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.text,
  },
  tradeSideBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  tradeSideText: {
    fontSize: 12,
    fontWeight: '600' as const,
  },
  tradeValue: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text,
  },
  tradeDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  tradeDetailItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  tradeDetailText: {
    fontSize: 12,
    color: Colors.textMuted,
  },
});
