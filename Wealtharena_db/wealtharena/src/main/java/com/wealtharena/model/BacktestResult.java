package com.wealtharena.model;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;
import java.util.Map;

public class BacktestResult {
    
    private String strategyName;
    private LocalDate startDate;
    private LocalDate endDate;
    private BigDecimal initialCapital;
    private BigDecimal finalCapital;
    private BigDecimal totalReturn;
    private BigDecimal annualizedReturn;
    private BigDecimal maxDrawdown;
    private BigDecimal sharpeRatio;
    private BigDecimal volatility;
    private BigDecimal winRate;
    private int totalTrades;
    private int winningTrades;
    private int losingTrades;
    private BigDecimal averageWin;
    private BigDecimal averageLoss;
    private BigDecimal profitFactor;
    private List<Trade> trades;
    private List<PortfolioSnapshot> portfolioSnapshots;
    private Map<String, Object> performanceMetrics;
    private LocalDate calculatedAt;
    
    // Constructors
    public BacktestResult() {
        this.calculatedAt = LocalDate.now();
    }
    
    public BacktestResult(String strategyName, LocalDate startDate, LocalDate endDate, 
                        BigDecimal initialCapital, BigDecimal finalCapital) {
        this();
        this.strategyName = strategyName;
        this.startDate = startDate;
        this.endDate = endDate;
        this.initialCapital = initialCapital;
        this.finalCapital = finalCapital;
        this.totalReturn = finalCapital.subtract(initialCapital).divide(initialCapital, 4, java.math.RoundingMode.HALF_UP);
    }
    
    // Inner class for individual trades
    public static class Trade {
        private LocalDate date;
        private String symbol;
        private String action; // BUY, SELL
        private BigDecimal quantity;
        private BigDecimal price;
        private BigDecimal value;
        private BigDecimal commission;
        private BigDecimal pnl;
        private BigDecimal cumulativePnl;
        
        public Trade() {}
        
        public Trade(LocalDate date, String symbol, String action, BigDecimal quantity, 
                    BigDecimal price, BigDecimal value, BigDecimal commission) {
            this.date = date;
            this.symbol = symbol;
            this.action = action;
            this.quantity = quantity;
            this.price = price;
            this.value = value;
            this.commission = commission;
        }
        
        // Getters and Setters
        public LocalDate getDate() { return date; }
        public void setDate(LocalDate date) { this.date = date; }
        
        public String getSymbol() { return symbol; }
        public void setSymbol(String symbol) { this.symbol = symbol; }
        
        public String getAction() { return action; }
        public void setAction(String action) { this.action = action; }
        
        public BigDecimal getQuantity() { return quantity; }
        public void setQuantity(BigDecimal quantity) { this.quantity = quantity; }
        
        public BigDecimal getPrice() { return price; }
        public void setPrice(BigDecimal price) { this.price = price; }
        
        public BigDecimal getValue() { return value; }
        public void setValue(BigDecimal value) { this.value = value; }
        
        public BigDecimal getCommission() { return commission; }
        public void setCommission(BigDecimal commission) { this.commission = commission; }
        
        public BigDecimal getPnl() { return pnl; }
        public void setPnl(BigDecimal pnl) { this.pnl = pnl; }
        
        public BigDecimal getCumulativePnl() { return cumulativePnl; }
        public void setCumulativePnl(BigDecimal cumulativePnl) { this.cumulativePnl = cumulativePnl; }
    }
    
    // Inner class for portfolio snapshots
    public static class PortfolioSnapshot {
        private LocalDate date;
        private BigDecimal totalValue;
        private BigDecimal cash;
        private Map<String, BigDecimal> positions; // symbol -> quantity
        private Map<String, BigDecimal> prices; // symbol -> price
        private BigDecimal dailyReturn;
        private BigDecimal cumulativeReturn;
        
        public PortfolioSnapshot() {}
        
        public PortfolioSnapshot(LocalDate date, BigDecimal totalValue, BigDecimal cash) {
            this.date = date;
            this.totalValue = totalValue;
            this.cash = cash;
        }
        
        // Getters and Setters
        public LocalDate getDate() { return date; }
        public void setDate(LocalDate date) { this.date = date; }
        
        public BigDecimal getTotalValue() { return totalValue; }
        public void setTotalValue(BigDecimal totalValue) { this.totalValue = totalValue; }
        
        public BigDecimal getCash() { return cash; }
        public void setCash(BigDecimal cash) { this.cash = cash; }
        
        public Map<String, BigDecimal> getPositions() { return positions; }
        public void setPositions(Map<String, BigDecimal> positions) { this.positions = positions; }
        
        public Map<String, BigDecimal> getPrices() { return prices; }
        public void setPrices(Map<String, BigDecimal> prices) { this.prices = prices; }
        
        public BigDecimal getDailyReturn() { return dailyReturn; }
        public void setDailyReturn(BigDecimal dailyReturn) { this.dailyReturn = dailyReturn; }
        
        public BigDecimal getCumulativeReturn() { return cumulativeReturn; }
        public void setCumulativeReturn(BigDecimal cumulativeReturn) { this.cumulativeReturn = cumulativeReturn; }
    }
    
    // Getters and Setters
    public String getStrategyName() { return strategyName; }
    public void setStrategyName(String strategyName) { this.strategyName = strategyName; }
    
    public LocalDate getStartDate() { return startDate; }
    public void setStartDate(LocalDate startDate) { this.startDate = startDate; }
    
    public LocalDate getEndDate() { return endDate; }
    public void setEndDate(LocalDate endDate) { this.endDate = endDate; }
    
    public BigDecimal getInitialCapital() { return initialCapital; }
    public void setInitialCapital(BigDecimal initialCapital) { this.initialCapital = initialCapital; }
    
    public BigDecimal getFinalCapital() { return finalCapital; }
    public void setFinalCapital(BigDecimal finalCapital) { this.finalCapital = finalCapital; }
    
    public BigDecimal getTotalReturn() { return totalReturn; }
    public void setTotalReturn(BigDecimal totalReturn) { this.totalReturn = totalReturn; }
    
    public BigDecimal getAnnualizedReturn() { return annualizedReturn; }
    public void setAnnualizedReturn(BigDecimal annualizedReturn) { this.annualizedReturn = annualizedReturn; }
    
    public BigDecimal getMaxDrawdown() { return maxDrawdown; }
    public void setMaxDrawdown(BigDecimal maxDrawdown) { this.maxDrawdown = maxDrawdown; }
    
    public BigDecimal getSharpeRatio() { return sharpeRatio; }
    public void setSharpeRatio(BigDecimal sharpeRatio) { this.sharpeRatio = sharpeRatio; }
    
    public BigDecimal getVolatility() { return volatility; }
    public void setVolatility(BigDecimal volatility) { this.volatility = volatility; }
    
    public BigDecimal getWinRate() { return winRate; }
    public void setWinRate(BigDecimal winRate) { this.winRate = winRate; }
    
    public int getTotalTrades() { return totalTrades; }
    public void setTotalTrades(int totalTrades) { this.totalTrades = totalTrades; }
    
    public int getWinningTrades() { return winningTrades; }
    public void setWinningTrades(int winningTrades) { this.winningTrades = winningTrades; }
    
    public int getLosingTrades() { return losingTrades; }
    public void setLosingTrades(int losingTrades) { this.losingTrades = losingTrades; }
    
    public BigDecimal getAverageWin() { return averageWin; }
    public void setAverageWin(BigDecimal averageWin) { this.averageWin = averageWin; }
    
    public BigDecimal getAverageLoss() { return averageLoss; }
    public void setAverageLoss(BigDecimal averageLoss) { this.averageLoss = averageLoss; }
    
    public BigDecimal getProfitFactor() { return profitFactor; }
    public void setProfitFactor(BigDecimal profitFactor) { this.profitFactor = profitFactor; }
    
    public List<Trade> getTrades() { return trades; }
    public void setTrades(List<Trade> trades) { this.trades = trades; }
    
    public List<PortfolioSnapshot> getPortfolioSnapshots() { return portfolioSnapshots; }
    public void setPortfolioSnapshots(List<PortfolioSnapshot> portfolioSnapshots) { this.portfolioSnapshots = portfolioSnapshots; }
    
    public Map<String, Object> getPerformanceMetrics() { return performanceMetrics; }
    public void setPerformanceMetrics(Map<String, Object> performanceMetrics) { this.performanceMetrics = performanceMetrics; }
    
    public LocalDate getCalculatedAt() { return calculatedAt; }
    public void setCalculatedAt(LocalDate calculatedAt) { this.calculatedAt = calculatedAt; }
}
