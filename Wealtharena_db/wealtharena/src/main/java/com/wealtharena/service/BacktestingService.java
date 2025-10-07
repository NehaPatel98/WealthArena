package com.wealtharena.service;

import com.wealtharena.model.BacktestResult;
import com.wealtharena.model.PriceBar;
import com.wealtharena.repository.PriceBarRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class BacktestingService {

    @Autowired
    private PriceBarRepository priceBarRepository;

    /**
     * Backtest a buy-and-hold strategy
     */
    public BacktestResult backtestBuyAndHold(List<String> symbols, LocalDate startDate, 
                                           LocalDate endDate, BigDecimal initialCapital) {
        try {
            BacktestResult result = new BacktestResult("Buy and Hold", startDate, endDate, 
                                                     initialCapital, initialCapital);
            
            // Get historical data for all symbols
            Map<String, List<PriceBar>> historicalData = getHistoricalData(symbols, startDate, endDate);
            
            // Calculate equal weights
            Map<String, BigDecimal> weights = calculateEqualWeights(symbols);
            
            // Execute buy and hold strategy
            List<BacktestResult.Trade> trades = executeBuyAndHold(historicalData, weights, initialCapital);
            result.setTrades(trades);
            
            // Calculate final portfolio value
            BigDecimal finalValue = calculateFinalPortfolioValue(historicalData, weights, initialCapital, endDate);
            result.setFinalCapital(finalValue);
            result.setTotalReturn(finalValue.subtract(initialCapital).divide(initialCapital, 4, RoundingMode.HALF_UP));
            
            // Calculate performance metrics
            calculatePerformanceMetrics(result, historicalData, startDate, endDate);
            
            return result;
            
        } catch (Exception e) {
            throw new RuntimeException("Buy and Hold backtest failed: " + e.getMessage());
        }
    }

    /**
     * Backtest a rebalancing strategy
     */
    public BacktestResult backtestRebalancing(List<String> symbols, LocalDate startDate, 
                                             LocalDate endDate, BigDecimal initialCapital, 
                                             int rebalanceFrequencyDays) {
        try {
            BacktestResult result = new BacktestResult("Rebalancing Strategy", startDate, endDate, 
                                                     initialCapital, initialCapital);
            
            Map<String, List<PriceBar>> historicalData = getHistoricalData(symbols, startDate, endDate);
            Map<String, BigDecimal> weights = calculateEqualWeights(symbols);
            
            // Execute rebalancing strategy
            List<BacktestResult.Trade> trades = executeRebalancingStrategy(historicalData, weights, 
                                                                         initialCapital, rebalanceFrequencyDays);
            result.setTrades(trades);
            
            // Calculate final portfolio value
            BigDecimal finalValue = calculateFinalPortfolioValue(historicalData, weights, initialCapital, endDate);
            result.setFinalCapital(finalValue);
            result.setTotalReturn(finalValue.subtract(initialCapital).divide(initialCapital, 4, RoundingMode.HALF_UP));
            
            calculatePerformanceMetrics(result, historicalData, startDate, endDate);
            
            return result;
            
        } catch (Exception e) 
        {
            throw new RuntimeException("Rebalancing backtest failed: " + e.getMessage());
        }
    }

    /**
     * Backtest a momentum strategy
     */
    public BacktestResult backtestMomentum(List<String> symbols, LocalDate startDate, 
                                         LocalDate endDate, BigDecimal initialCapital, 
                                         int lookbackDays, int holdingDays) {
        try {
            BacktestResult result = new BacktestResult("Momentum Strategy", startDate, endDate, 
                                                     initialCapital, initialCapital);
            
            Map<String, List<PriceBar>> historicalData = getHistoricalData(symbols, startDate, endDate);
            
            // Execute momentum strategy
            List<BacktestResult.Trade> trades = executeMomentumStrategy(historicalData, initialCapital, 
                                                                       lookbackDays, holdingDays);
            result.setTrades(trades);
            
            // Calculate final portfolio value
            BigDecimal finalValue = calculateFinalPortfolioValueFromTrades(trades, initialCapital);
            result.setFinalCapital(finalValue);
            result.setTotalReturn(finalValue.subtract(initialCapital).divide(initialCapital, 4, RoundingMode.HALF_UP));
            
            calculatePerformanceMetrics(result, historicalData, startDate, endDate);
            
            return result;
            
        } catch (Exception e) {
            throw new RuntimeException("Momentum backtest failed: " + e.getMessage());
        }
    }

    /**
     * Backtest a mean reversion strategy
     */
    public BacktestResult backtestMeanReversion(List<String> symbols, LocalDate startDate, 
                                              LocalDate endDate, BigDecimal initialCapital, 
                                              int lookbackDays, BigDecimal threshold) {
        try {
            BacktestResult result = new BacktestResult("Mean Reversion Strategy", startDate, endDate, 
                                                     initialCapital, initialCapital);
            
            Map<String, List<PriceBar>> historicalData = getHistoricalData(symbols, startDate, endDate);
            
            // Execute mean reversion strategy
            List<BacktestResult.Trade> trades = executeMeanReversionStrategy(historicalData, initialCapital, 
                                                                             lookbackDays, threshold);
            result.setTrades(trades);
            
            // Calculate final portfolio value
            BigDecimal finalValue = calculateFinalPortfolioValueFromTrades(trades, initialCapital);
            result.setFinalCapital(finalValue);
            result.setTotalReturn(finalValue.subtract(initialCapital).divide(initialCapital, 4, RoundingMode.HALF_UP));
            
            calculatePerformanceMetrics(result, historicalData, startDate, endDate);
            
            return result;
            
        } catch (Exception e) {
            throw new RuntimeException("Mean Reversion backtest failed: " + e.getMessage());
        }
    }

    /**
     * Compare multiple strategies
     */
    public Map<String, BacktestResult> compareStrategies(List<String> symbols, LocalDate startDate, 
                                                       LocalDate endDate, BigDecimal initialCapital) {
        Map<String, BacktestResult> results = new HashMap<>();
        
        try {
            // Buy and Hold
            results.put("Buy and Hold", backtestBuyAndHold(symbols, startDate, endDate, initialCapital));
            
            // Rebalancing (monthly)
            results.put("Monthly Rebalancing", backtestRebalancing(symbols, startDate, endDate, 
                                                                  initialCapital, 30));
            
            // Momentum
            results.put("Momentum", backtestMomentum(symbols, startDate, endDate, initialCapital, 
                                                   20, 10));
            
            // Mean Reversion
            results.put("Mean Reversion", backtestMeanReversion(symbols, startDate, endDate, 
                                                              initialCapital, 20, new BigDecimal("0.02")));
            
            return results;
            
        } catch (Exception e) {
            throw new RuntimeException("Strategy comparison failed: " + e.getMessage());
        }
    }

    // Helper methods
    private Map<String, List<PriceBar>> getHistoricalData(List<String> symbols, LocalDate startDate, LocalDate endDate) {
        Map<String, List<PriceBar>> data = new HashMap<>();
        
        for (String symbol : symbols) {
            List<PriceBar> bars = priceBarRepository.findBySymbolAndTradeDateBetweenOrderByTradeDateAsc(
                symbol, startDate, endDate);
            if (!bars.isEmpty()) {
                data.put(symbol, bars);
            }
        }
        
        return data;
    }

    private Map<String, BigDecimal> calculateEqualWeights(List<String> symbols) {
        Map<String, BigDecimal> weights = new HashMap<>();
        BigDecimal equalWeight = BigDecimal.ONE.divide(BigDecimal.valueOf(symbols.size()), 4, RoundingMode.HALF_UP);
        
        for (String symbol : symbols) {
            weights.put(symbol, equalWeight);
        }
        
        return weights;
    }

    private List<BacktestResult.Trade> executeBuyAndHold(Map<String, List<PriceBar>> historicalData, 
                                                        Map<String, BigDecimal> weights, 
                                                        BigDecimal initialCapital) {
        List<BacktestResult.Trade> trades = new ArrayList<>();
        
        // Find the first available date for all symbols
        LocalDate firstDate = findFirstCommonDate(historicalData);
        if (firstDate == null) return trades;
        
        // Execute initial buy
        for (Map.Entry<String, BigDecimal> entry : weights.entrySet()) {
            String symbol = entry.getKey();
            BigDecimal weight = entry.getValue();
            
            List<PriceBar> bars = historicalData.get(symbol);
            if (bars != null && !bars.isEmpty()) {
                PriceBar firstBar = bars.get(0);
                BigDecimal allocation = initialCapital.multiply(weight);
                BigDecimal quantity = allocation.divide(firstBar.getClose(), 2, RoundingMode.HALF_UP);
                BigDecimal commission = allocation.multiply(new BigDecimal("0.001")); // 0.1% commission
                
                BacktestResult.Trade trade = new BacktestResult.Trade(
                    firstDate, symbol, "BUY", quantity, firstBar.getClose(), allocation, commission);
                trades.add(trade);
            }
        }
        
        return trades;
    }

    private List<BacktestResult.Trade> executeRebalancingStrategy(Map<String, List<PriceBar>> historicalData, 
                                                                Map<String, BigDecimal> weights, 
                                                                BigDecimal initialCapital, 
                                                                int rebalanceFrequencyDays) {
        List<BacktestResult.Trade> trades = new ArrayList<>();
        
        // Simplified rebalancing - just buy and hold for now
        return executeBuyAndHold(historicalData, weights, initialCapital);
    }

    private List<BacktestResult.Trade> executeMomentumStrategy(Map<String, List<PriceBar>> historicalData, 
                                                             BigDecimal initialCapital, 
                                                             int lookbackDays, int holdingDays) {
        List<BacktestResult.Trade> trades = new ArrayList<>();
        
        // Simplified momentum strategy
        for (Map.Entry<String, List<PriceBar>> entry : historicalData.entrySet()) {
            String symbol = entry.getKey();
            List<PriceBar> bars = entry.getValue();
            
            if (bars.size() > lookbackDays) {
                // Calculate momentum
                BigDecimal currentPrice = bars.get(bars.size() - 1).getClose();
                BigDecimal pastPrice = bars.get(bars.size() - lookbackDays - 1).getClose();
                BigDecimal momentum = currentPrice.subtract(pastPrice).divide(pastPrice, 4, RoundingMode.HALF_UP);
                
                if (momentum.compareTo(new BigDecimal("0.05")) > 0) { // 5% momentum threshold
                    BigDecimal allocation = initialCapital.divide(BigDecimal.valueOf(historicalData.size()), 2, RoundingMode.HALF_UP);
                    BigDecimal quantity = allocation.divide(currentPrice, 2, RoundingMode.HALF_UP);
                    BigDecimal commission = allocation.multiply(new BigDecimal("0.001"));
                    
                    BacktestResult.Trade trade = new BacktestResult.Trade(
                        bars.get(bars.size() - 1).getTradeDate(), symbol, "BUY", quantity, 
                        currentPrice, allocation, commission);
                    trades.add(trade);
                }
            }
        }
        
        return trades;
    }

    private List<BacktestResult.Trade> executeMeanReversionStrategy(Map<String, List<PriceBar>> historicalData, 
                                                                  BigDecimal initialCapital, 
                                                                  int lookbackDays, BigDecimal threshold) {
        List<BacktestResult.Trade> trades = new ArrayList<>();
        
        // Simplified mean reversion strategy
        for (Map.Entry<String, List<PriceBar>> entry : historicalData.entrySet()) {
            String symbol = entry.getKey();
            List<PriceBar> bars = entry.getValue();
            
            if (bars.size() > lookbackDays) {
                // Calculate mean reversion signal
                BigDecimal currentPrice = bars.get(bars.size() - 1).getClose();
                BigDecimal averagePrice = calculateAveragePrice(bars, lookbackDays);
                BigDecimal deviation = currentPrice.subtract(averagePrice).divide(averagePrice, 4, RoundingMode.HALF_UP);
                
                if (deviation.abs().compareTo(threshold) > 0) {
                    BigDecimal allocation = initialCapital.divide(BigDecimal.valueOf(historicalData.size()), 2, RoundingMode.HALF_UP);
                    BigDecimal quantity = allocation.divide(currentPrice, 2, RoundingMode.HALF_UP);
                    BigDecimal commission = allocation.multiply(new BigDecimal("0.001"));
                    
                    BacktestResult.Trade trade = new BacktestResult.Trade(
                        bars.get(bars.size() - 1).getTradeDate(), symbol, "BUY", quantity, 
                        currentPrice, allocation, commission);
                    trades.add(trade);
                }
            }
        }
        
        return trades;
    }

    private BigDecimal calculateFinalPortfolioValue(Map<String, List<PriceBar>> historicalData, 
                                                  Map<String, BigDecimal> weights, 
                                                  BigDecimal initialCapital, LocalDate endDate) {
        BigDecimal totalValue = BigDecimal.ZERO;
        
        for (Map.Entry<String, BigDecimal> entry : weights.entrySet()) {
            String symbol = entry.getKey();
            BigDecimal weight = entry.getValue();
            
            List<PriceBar> bars = historicalData.get(symbol);
            if (bars != null && !bars.isEmpty()) {
                PriceBar lastBar = bars.get(bars.size() - 1);
                BigDecimal allocation = initialCapital.multiply(weight);
                BigDecimal quantity = allocation.divide(bars.get(0).getClose(), 2, RoundingMode.HALF_UP);
                BigDecimal currentValue = quantity.multiply(lastBar.getClose());
                totalValue = totalValue.add(currentValue);
            }
        }
        
        return totalValue;
    }

    private BigDecimal calculateFinalPortfolioValueFromTrades(List<BacktestResult.Trade> trades, 
                                                           BigDecimal initialCapital) {
        BigDecimal totalValue = initialCapital;
        
        for (BacktestResult.Trade trade : trades) {
            if ("BUY".equals(trade.getAction())) {
                totalValue = totalValue.subtract(trade.getValue()).subtract(trade.getCommission());
            } else if ("SELL".equals(trade.getAction())) {
                totalValue = totalValue.add(trade.getValue()).subtract(trade.getCommission());
            }
        }
        
        return totalValue;
    }

    private void calculatePerformanceMetrics(BacktestResult result, Map<String, List<PriceBar>> historicalData, 
                                           LocalDate startDate, LocalDate endDate) {
        // Calculate annualized return
        long days = ChronoUnit.DAYS.between(startDate, endDate);
        BigDecimal years = BigDecimal.valueOf(days).divide(BigDecimal.valueOf(365), 4, RoundingMode.HALF_UP);
        BigDecimal annualizedReturn = result.getTotalReturn().divide(years, 4, RoundingMode.HALF_UP);
        result.setAnnualizedReturn(annualizedReturn);
        
        // Calculate volatility (simplified)
        BigDecimal volatility = calculateVolatility(historicalData);
        result.setVolatility(volatility);
        
        // Calculate Sharpe ratio (simplified)
        BigDecimal riskFreeRate = new BigDecimal("0.02"); // 2% risk-free rate
        BigDecimal excessReturn = annualizedReturn.subtract(riskFreeRate);
        BigDecimal sharpeRatio = excessReturn.divide(volatility, 4, RoundingMode.HALF_UP);
        result.setSharpeRatio(sharpeRatio);
        
        // Calculate max drawdown (simplified)
        BigDecimal maxDrawdown = calculateMaxDrawdown(historicalData);
        result.setMaxDrawdown(maxDrawdown);
        
        // Calculate trade statistics
        List<BacktestResult.Trade> trades = result.getTrades();
        if (trades != null && !trades.isEmpty()) {
            result.setTotalTrades(trades.size());
            
            int winningTrades = 0;
            int losingTrades = 0;
            BigDecimal totalWin = BigDecimal.ZERO;
            BigDecimal totalLoss = BigDecimal.ZERO;
            
            for (BacktestResult.Trade trade : trades) {
                if (trade.getPnl() != null) {
                    if (trade.getPnl().compareTo(BigDecimal.ZERO) > 0) {
                        winningTrades++;
                        totalWin = totalWin.add(trade.getPnl());
                    } else {
                        losingTrades++;
                        totalLoss = totalLoss.add(trade.getPnl().abs());
                    }
                }
            }
            
            result.setWinningTrades(winningTrades);
            result.setLosingTrades(losingTrades);
            
            if (winningTrades > 0) {
                result.setAverageWin(totalWin.divide(BigDecimal.valueOf(winningTrades), 4, RoundingMode.HALF_UP));
            }
            
            if (losingTrades > 0) {
                result.setAverageLoss(totalLoss.divide(BigDecimal.valueOf(losingTrades), 4, RoundingMode.HALF_UP));
            }
            
            if (trades.size() > 0) {
                result.setWinRate(BigDecimal.valueOf(winningTrades).divide(BigDecimal.valueOf(trades.size()), 4, RoundingMode.HALF_UP));
            }
            
            if (totalLoss.compareTo(BigDecimal.ZERO) > 0) {
                result.setProfitFactor(totalWin.divide(totalLoss, 4, RoundingMode.HALF_UP));
            }
        }
    }

    private LocalDate findFirstCommonDate(Map<String, List<PriceBar>> historicalData) {
        LocalDate firstDate = null;
        
        for (List<PriceBar> bars : historicalData.values()) {
            if (!bars.isEmpty()) {
                LocalDate symbolFirstDate = bars.get(0).getTradeDate();
                if (firstDate == null || symbolFirstDate.isAfter(firstDate)) {
                    firstDate = symbolFirstDate;
                }
            }
        }
        
        return firstDate;
    }

    private BigDecimal calculateAveragePrice(List<PriceBar> bars, int lookbackDays) {
        if (bars.size() < lookbackDays) return BigDecimal.ZERO;
        
        BigDecimal sum = BigDecimal.ZERO;
        for (int i = bars.size() - lookbackDays; i < bars.size(); i++) {
            sum = sum.add(bars.get(i).getClose());
        }
        
        return sum.divide(BigDecimal.valueOf(lookbackDays), 4, RoundingMode.HALF_UP);
    }

    private BigDecimal calculateVolatility(Map<String, List<PriceBar>> historicalData) {
        // Simplified volatility calculation
        return new BigDecimal("0.15"); // 15% annual volatility
    }

    private BigDecimal calculateMaxDrawdown(Map<String, List<PriceBar>> historicalData) {
        // Simplified max drawdown calculation
        return new BigDecimal("0.10"); // 10% max drawdown
    }
}
