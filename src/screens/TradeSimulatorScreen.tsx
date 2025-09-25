import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { colors } from '../theme/colors';

interface Stock {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: string;
}

interface Order {
  id: string;
  symbol: string;
  type: 'market' | 'limit' | 'stop';
  side: 'buy' | 'sell';
  quantity: number;
  price?: number;
  status: 'pending' | 'filled' | 'cancelled';
  timestamp: string;
}

interface Position {
  symbol: string;
  quantity: number;
  avgPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
}

interface TradeSimulatorScreenProps {
  userProfile: {
    experience: 'beginner' | 'intermediate' | 'advanced';
  };
  onBack: () => void;
}

const TradeSimulatorScreen: React.FC<TradeSimulatorScreenProps> = ({ userProfile, onBack }) => {
  const [activeTab, setActiveTab] = useState<'market' | 'orders' | 'positions' | 'history'>('market');
  const [selectedStock, setSelectedStock] = useState<Stock | null>(null);
  const [orderType, setOrderType] = useState<'market' | 'limit' | 'stop'>('market');
  const [orderSide, setOrderSide] = useState<'buy' | 'sell'>('buy');
  const [orderQuantity, setOrderQuantity] = useState('');
  const [orderPrice, setOrderPrice] = useState('');
  const [availableCash, setAvailableCash] = useState(100000);
  const [positions, setPositions] = useState<Position[]>([]);
  const [orders, setOrders] = useState<Order[]>([]);
  
  const isDarkMode = true;
  const c = colors[isDarkMode ? 'dark' : 'light'];

  const marketStocks: Stock[] = [
    {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      price: 175.43,
      change: 2.15,
      changePercent: 1.24,
      volume: 45678900,
      marketCap: '2.7T'
    },
    {
      symbol: 'MSFT',
      name: 'Microsoft Corporation',
      price: 378.85,
      change: -1.23,
      changePercent: -0.32,
      volume: 23456700,
      marketCap: '2.8T'
    },
    {
      symbol: 'GOOGL',
      name: 'Alphabet Inc.',
      price: 142.56,
      change: 3.45,
      changePercent: 2.48,
      volume: 18923400,
      marketCap: '1.8T'
    },
    {
      symbol: 'AMZN',
      name: 'Amazon.com Inc.',
      price: 155.78,
      change: -0.89,
      changePercent: -0.57,
      volume: 32145600,
      marketCap: '1.6T'
    },
    {
      symbol: 'TSLA',
      name: 'Tesla Inc.',
      price: 248.42,
      change: 8.76,
      changePercent: 3.66,
      volume: 67890100,
      marketCap: '790B'
    }
  ];

  const handleStockSelect = (stock: Stock) => {
    setSelectedStock(stock);
  };

  const handleOrderSubmit = () => {
    if (!selectedStock || !orderQuantity) {
      Alert.alert('Error', 'Please select a stock and enter quantity');
      return;
    }

    const quantity = parseInt(orderQuantity, 10);
    const price = orderPrice ? parseFloat(orderPrice) : selectedStock.price;
    const totalCost = quantity * price;

    if (orderSide === 'buy' && totalCost > availableCash) {
      Alert.alert('Error', 'Insufficient funds');
      return;
    }

    const newOrder: Order = {
      id: Date.now().toString(),
      symbol: selectedStock.symbol,
      type: orderType,
      side: orderSide,
      quantity,
      price: orderType !== 'market' ? price : undefined,
      status: 'pending',
      timestamp: new Date().toISOString()
    };

    setOrders(prev => [...prev, newOrder]);
    
    // Simulate order execution
    setTimeout(() => {
      setOrders(prev => prev.map(order => 
        order.id === newOrder.id ? { ...order, status: 'filled' } : order
      ));
      
      // Update positions
      updatePositions(newOrder, selectedStock);
      
      // Update cash
      if (orderSide === 'buy') {
        setAvailableCash(prev => prev - totalCost);
      } else {
        setAvailableCash(prev => prev + totalCost);
      }
    }, 2000);

    // Reset form
    setOrderQuantity('');
    setOrderPrice('');
  };

  const updatePositions = (order: Order, stock: Stock) => {
    setPositions(prev => {
      const existingPosition = prev.find(p => p.symbol === order.symbol);
      
      if (existingPosition) {
        if (order.side === 'buy') {
          const newQuantity = existingPosition.quantity + order.quantity;
          const newAvgPrice = ((existingPosition.avgPrice * existingPosition.quantity) + 
                              (stock.price * order.quantity)) / newQuantity;
          return prev.map(p => 
            p.symbol === order.symbol 
              ? { ...p, quantity: newQuantity, avgPrice: newAvgPrice }
              : p
          );
        } else {
          const newQuantity = existingPosition.quantity - order.quantity;
          if (newQuantity <= 0) {
            return prev.filter(p => p.symbol !== order.symbol);
          }
          return prev.map(p => 
            p.symbol === order.symbol 
              ? { ...p, quantity: newQuantity }
              : p
          );
        }
      } else if (order.side === 'buy') {
        return [...prev, {
          symbol: order.symbol,
          quantity: order.quantity,
          avgPrice: stock.price,
          currentPrice: stock.price,
          pnl: 0,
          pnlPercent: 0
        }];
      }
      
      return prev;
    });
  };

  const renderMarket = () => (
    <View style={styles.tabContent}>
      <Text style={[styles.sectionTitle, { color: c.text }]}>Market Data</Text>
      
      <View style={styles.stocksList}>
        {marketStocks.map((stock) => (
          <TouchableOpacity
            key={stock.symbol}
            style={[
              styles.stockCard,
              { backgroundColor: c.surface },
              selectedStock?.symbol === stock.symbol && { borderColor: c.primary, borderWidth: 2 }
            ]}
            onPress={() => handleStockSelect(stock)}
          >
            <View style={styles.stockHeader}>
              <View style={styles.stockInfo}>
                <Text style={[styles.stockSymbol, { color: c.text }]}>{stock.symbol}</Text>
                <Text style={[styles.stockName, { color: c.textMuted }]}>{stock.name}</Text>
              </View>
              <View style={styles.stockPrice}>
                <Text style={[styles.price, { color: c.text }]}>${stock.price.toFixed(2)}</Text>
                <View style={[
                  styles.changeContainer,
                  { backgroundColor: stock.change >= 0 ? c.success + '20' : c.danger + '20' }
                ]}>
                  <Icon 
                    name={stock.change >= 0 ? 'trending-up' : 'trending-down'} 
                    size={12} 
                    color={stock.change >= 0 ? c.success : c.danger} 
                  />
                  <Text style={[
                    styles.change,
                    { color: stock.change >= 0 ? c.success : c.danger }
                  ]}>
                    {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)} ({stock.changePercent.toFixed(2)}%)
                  </Text>
                </View>
              </View>
            </View>
            
            <View style={styles.stockDetails}>
              <View style={styles.detailItem}>
                <Text style={[styles.detailLabel, { color: c.textMuted }]}>Volume</Text>
                <Text style={[styles.detailValue, { color: c.text }]}>
                  {stock.volume.toLocaleString()}
                </Text>
              </View>
              <View style={styles.detailItem}>
                <Text style={[styles.detailLabel, { color: c.textMuted }]}>Market Cap</Text>
                <Text style={[styles.detailValue, { color: c.text }]}>{stock.marketCap}</Text>
              </View>
            </View>
          </TouchableOpacity>
        ))}
      </View>
      
      {selectedStock && (
        <View style={[styles.orderForm, { backgroundColor: c.surface }]}>
          <Text style={[styles.formTitle, { color: c.text }]}>Place Order</Text>
          
          <View style={styles.orderTypeSelector}>
            {[
              { id: 'market', label: 'Market', icon: 'flash' },
              { id: 'limit', label: 'Limit', icon: 'target' },
              { id: 'stop', label: 'Stop', icon: 'stop' }
            ].map((type) => (
              <TouchableOpacity
                key={type.id}
                style={[
                  styles.orderTypeButton,
                  { backgroundColor: orderType === type.id ? c.primary : c.background },
                  { borderColor: c.border }
                ]}
                onPress={() => setOrderType(type.id as any)}
              >
                <Icon 
                  name={type.icon} 
                  size={16} 
                  color={orderType === type.id ? c.background : c.text} 
                />
                <Text style={[
                  styles.orderTypeText,
                  { color: orderType === type.id ? c.background : c.text }
                ]}>
                  {type.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
          
          <View style={styles.orderSideSelector}>
            <TouchableOpacity
              style={[
                styles.sideButton,
                { backgroundColor: orderSide === 'buy' ? c.success : c.background },
                { borderColor: c.border }
              ]}
              onPress={() => setOrderSide('buy')}
            >
              <Icon name="arrow-up" size={16} color={orderSide === 'buy' ? c.background : c.success} />
              <Text style={[
                styles.sideText,
                { color: orderSide === 'buy' ? c.background : c.success }
              ]}>
                Buy
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[
                styles.sideButton,
                { backgroundColor: orderSide === 'sell' ? c.danger : c.background },
                { borderColor: c.border }
              ]}
              onPress={() => setOrderSide('sell')}
            >
              <Icon name="arrow-down" size={16} color={orderSide === 'sell' ? c.background : c.danger} />
              <Text style={[
                styles.sideText,
                { color: orderSide === 'sell' ? c.background : c.danger }
              ]}>
                Sell
              </Text>
            </TouchableOpacity>
          </View>
          
          <View style={styles.orderInputs}>
            <View style={styles.inputGroup}>
              <Text style={[styles.inputLabel, { color: c.text }]}>Quantity</Text>
              <TextInput
                style={[styles.input, { backgroundColor: c.background, borderColor: c.border, color: c.text }]}
                value={orderQuantity}
                onChangeText={setOrderQuantity}
                placeholder="Enter quantity"
                placeholderTextColor={c.textMuted}
                keyboardType="numeric"
              />
            </View>
            
            {orderType !== 'market' && (
              <View style={styles.inputGroup}>
                <Text style={[styles.inputLabel, { color: c.text }]}>Price</Text>
                <TextInput
                  style={[styles.input, { backgroundColor: c.background, borderColor: c.border, color: c.text }]}
                  value={orderPrice}
                  onChangeText={setOrderPrice}
                  placeholder="Enter price"
                  placeholderTextColor={c.textMuted}
                  keyboardType="numeric"
                />
              </View>
            )}
          </View>
          
          <TouchableOpacity
            style={[styles.submitButton, { backgroundColor: c.primary }]}
            onPress={handleOrderSubmit}
          >
            <Text style={[styles.submitText, { color: c.background }]}>
              Place {orderSide.toUpperCase()} Order
            </Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );

  const renderPositions = () => (
    <View style={styles.tabContent}>
      <Text style={[styles.sectionTitle, { color: c.text }]}>Positions</Text>
      
      {positions.length > 0 ? (
        <View style={styles.positionsList}>
          {positions.map((position) => (
            <View key={position.symbol} style={[styles.positionCard, { backgroundColor: c.surface }]}>
              <View style={styles.positionHeader}>
                <Text style={[styles.positionSymbol, { color: c.text }]}>{position.symbol}</Text>
                <Text style={[styles.positionQuantity, { color: c.textMuted }]}>
                  {position.quantity} shares
                </Text>
              </View>
              
              <View style={styles.positionDetails}>
                <View style={styles.positionInfo}>
                  <Text style={[styles.positionLabel, { color: c.textMuted }]}>Avg Price</Text>
                  <Text style={[styles.positionValue, { color: c.text }]}>
                    ${position.avgPrice.toFixed(2)}
                  </Text>
                </View>
                <View style={styles.positionInfo}>
                  <Text style={[styles.positionLabel, { color: c.textMuted }]}>Current Price</Text>
                  <Text style={[styles.positionValue, { color: c.text }]}>
                    ${position.currentPrice.toFixed(2)}
                  </Text>
                </View>
                <View style={styles.positionInfo}>
                  <Text style={[styles.positionLabel, { color: c.textMuted }]}>P&L</Text>
                  <Text style={[
                    styles.positionPnl,
                    { color: position.pnl >= 0 ? c.success : c.danger }
                  ]}>
                    {position.pnl >= 0 ? '+' : ''}${position.pnl.toFixed(2)}
                  </Text>
                </View>
              </View>
            </View>
          ))}
        </View>
      ) : (
        <View style={styles.emptyState}>
          <Icon name="briefcase-outline" size={48} color={c.textMuted} />
          <Text style={[styles.emptyText, { color: c.text }]}>No Positions</Text>
          <Text style={[styles.emptySubtext, { color: c.textMuted }]}>
            Start trading to build your portfolio
          </Text>
        </View>
      )}
    </View>
  );

  const renderOrders = () => (
    <View style={styles.tabContent}>
      <Text style={[styles.sectionTitle, { color: c.text }]}>Orders</Text>
      
      {orders.length > 0 ? (
        <View style={styles.ordersList}>
          {orders.map((order) => (
            <View key={order.id} style={[styles.orderCard, { backgroundColor: c.surface }]}>
              <View style={styles.orderHeader}>
                <Text style={[styles.orderSymbol, { color: c.text }]}>{order.symbol}</Text>
                <View style={[
                  styles.statusBadge,
                  { backgroundColor: order.status === 'filled' ? c.success : 
                                   order.status === 'pending' ? c.warning : c.danger }
                ]}>
                  <Text style={[styles.statusText, { color: c.background }]}>
                    {order.status.toUpperCase()}
                  </Text>
                </View>
              </View>
              
              <View style={styles.orderDetails}>
                <Text style={[styles.orderInfo, { color: c.textMuted }]}>
                  {order.side.toUpperCase()} {order.quantity} shares @ {order.type.toUpperCase()}
                </Text>
                {order.price && (
                  <Text style={[styles.orderPrice, { color: c.text }]}>
                    ${order.price.toFixed(2)}
                  </Text>
                )}
              </View>
            </View>
          ))}
        </View>
      ) : (
        <View style={styles.emptyState}>
          <Icon name="clipboard-list-outline" size={48} color={c.textMuted} />
          <Text style={[styles.emptyText, { color: c.text }]}>No Orders</Text>
          <Text style={[styles.emptySubtext, { color: c.textMuted }]}>
            Place your first order to get started
          </Text>
        </View>
      )}
    </View>
  );

  return (
    <View style={[styles.container, { backgroundColor: c.background }]}>
      {/* Header */}
      <View style={[styles.header, { borderBottomColor: c.border }]}>
        <TouchableOpacity onPress={onBack} style={styles.backButton}>
          <Icon name="arrow-left" size={24} color={c.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: c.text }]}>Trade Simulator</Text>
        <View style={styles.headerActions}>
          <View style={[styles.cashDisplay, { backgroundColor: c.primary }]}>
            <Text style={[styles.cashText, { color: c.background }]}>
              ${availableCash.toLocaleString()}
            </Text>
          </View>
        </View>
      </View>

      {/* Tabs */}
      <View style={[styles.tabsContainer, { backgroundColor: c.surface }]}>
        {[
          { id: 'market', label: 'Market', icon: 'chart-line' },
          { id: 'orders', label: 'Orders', icon: 'clipboard-list' },
          { id: 'positions', label: 'Positions', icon: 'briefcase' },
          { id: 'history', label: 'History', icon: 'history' }
        ].map((tab) => (
          <TouchableOpacity
            key={tab.id}
            style={[
              styles.tab,
              activeTab === tab.id && { backgroundColor: c.primary }
            ]}
            onPress={() => setActiveTab(tab.id as any)}
          >
            <Icon 
              name={tab.icon} 
              size={20} 
              color={activeTab === tab.id ? c.background : c.textMuted} 
            />
            <Text style={[
              styles.tabText,
              { color: activeTab === tab.id ? c.background : c.textMuted }
            ]}>
              {tab.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Content */}
      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {activeTab === 'market' && renderMarket()}
        {activeTab === 'orders' && renderOrders()}
        {activeTab === 'positions' && renderPositions()}
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
  headerActions: {
    flexDirection: 'row',
    gap: 8,
  },
  cashDisplay: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  cashText: {
    fontSize: 14,
    fontWeight: '700',
  },
  tabsContainer: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    paddingVertical: 12,
    gap: 8,
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    gap: 8,
  },
  tabText: {
    fontSize: 14,
    fontWeight: '600',
  },
  content: {
    flex: 1,
    padding: 20,
  },
  tabContent: {
    flex: 1,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '700',
    marginBottom: 16,
  },
  stocksList: {
    gap: 12,
    marginBottom: 20,
  },
  stockCard: {
    padding: 16,
    borderRadius: 12,
  },
  stockHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  stockInfo: {
    flex: 1,
  },
  stockSymbol: {
    fontSize: 18,
    fontWeight: '700',
  },
  stockName: {
    fontSize: 14,
    marginTop: 2,
  },
  stockPrice: {
    alignItems: 'flex-end',
  },
  price: {
    fontSize: 18,
    fontWeight: '700',
  },
  changeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    marginTop: 4,
    gap: 4,
  },
  change: {
    fontSize: 12,
    fontWeight: '600',
  },
  stockDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  detailItem: {
    alignItems: 'center',
  },
  detailLabel: {
    fontSize: 12,
    marginBottom: 4,
  },
  detailValue: {
    fontSize: 14,
    fontWeight: '600',
  },
  orderForm: {
    padding: 20,
    borderRadius: 12,
  },
  formTitle: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 16,
  },
  orderTypeSelector: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 16,
  },
  orderTypeButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: 8,
    borderWidth: 1,
    gap: 8,
  },
  orderTypeText: {
    fontSize: 14,
    fontWeight: '600',
  },
  orderSideSelector: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  sideButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: 8,
    borderWidth: 1,
    gap: 8,
  },
  sideText: {
    fontSize: 14,
    fontWeight: '600',
  },
  orderInputs: {
    gap: 16,
    marginBottom: 20,
  },
  inputGroup: {
    gap: 8,
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: '600',
  },
  input: {
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 16,
  },
  submitButton: {
    paddingVertical: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  submitText: {
    fontSize: 16,
    fontWeight: '700',
  },
  positionsList: {
    gap: 12,
  },
  positionCard: {
    padding: 16,
    borderRadius: 12,
  },
  positionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  positionSymbol: {
    fontSize: 18,
    fontWeight: '700',
  },
  positionQuantity: {
    fontSize: 14,
  },
  positionDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  positionInfo: {
    alignItems: 'center',
  },
  positionLabel: {
    fontSize: 12,
    marginBottom: 4,
  },
  positionValue: {
    fontSize: 14,
    fontWeight: '600',
  },
  positionPnl: {
    fontSize: 14,
    fontWeight: '700',
  },
  ordersList: {
    gap: 12,
  },
  orderCard: {
    padding: 16,
    borderRadius: 12,
  },
  orderHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  orderSymbol: {
    fontSize: 16,
    fontWeight: '700',
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  statusText: {
    fontSize: 10,
    fontWeight: '700',
  },
  orderDetails: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  orderInfo: {
    fontSize: 14,
    flex: 1,
  },
  orderPrice: {
    fontSize: 14,
    fontWeight: '600',
  },
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 60,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: '600',
    marginTop: 16,
  },
  emptySubtext: {
    fontSize: 14,
    marginTop: 8,
    textAlign: 'center',
  },
});

export default TradeSimulatorScreen;
