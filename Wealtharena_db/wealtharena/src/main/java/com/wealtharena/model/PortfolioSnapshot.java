package com.wealtharena.model;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.Map;

public class PortfolioSnapshot {
    
    private Long portfolioId;
    private String portfolioName;
    private LocalDateTime timestamp;
    private BigDecimal totalValue;
    private BigDecimal cash;
    private BigDecimal totalCost;
    private BigDecimal totalPnl;
    private BigDecimal totalPnlPercent;
    private BigDecimal dayChange;
    private BigDecimal dayChangePercent;
    private Map<String, PositionSnapshot> positions;
    private Map<String, Object> performanceMetrics;
    
    // Constructors
    public PortfolioSnapshot() {
        this.timestamp = LocalDateTime.now();
    }
    
    public PortfolioSnapshot(Long portfolioId, String portfolioName, BigDecimal totalValue, 
                           BigDecimal cash, BigDecimal totalCost) {
        this();
        this.portfolioId = portfolioId;
        this.portfolioName = portfolioName;
        this.totalValue = totalValue;
        this.cash = cash;
        this.totalCost = totalCost;
        this.totalPnl = totalValue.subtract(totalCost);
        this.totalPnlPercent = totalCost.compareTo(BigDecimal.ZERO) > 0 ? 
            totalPnl.divide(totalCost, 4, java.math.RoundingMode.HALF_UP) : BigDecimal.ZERO;
    }
    
    // Inner class for position snapshots
    public static class PositionSnapshot {
        private String symbol;
        private BigDecimal quantity;
        private BigDecimal currentPrice;
        private BigDecimal marketValue;
        private BigDecimal costBasis;
        private BigDecimal pnl;
        private BigDecimal pnlPercent;
        private BigDecimal dayChange;
        private BigDecimal dayChangePercent;
        private BigDecimal weight;
        private LocalDateTime lastUpdated;
        
        public PositionSnapshot() {
            this.lastUpdated = LocalDateTime.now();
        }
        
        public PositionSnapshot(String symbol, BigDecimal quantity, BigDecimal currentPrice, 
                              BigDecimal costBasis) {
            this();
            this.symbol = symbol;
            this.quantity = quantity;
            this.currentPrice = currentPrice;
            this.costBasis = costBasis;
            this.marketValue = quantity.multiply(currentPrice);
            this.pnl = marketValue.subtract(costBasis);
            this.pnlPercent = costBasis.compareTo(BigDecimal.ZERO) > 0 ? 
                pnl.divide(costBasis, 4, java.math.RoundingMode.HALF_UP) : BigDecimal.ZERO;
        }
        
        // Getters and Setters
        public String getSymbol() { return symbol; }
        public void setSymbol(String symbol) { this.symbol = symbol; }
        
        public BigDecimal getQuantity() { return quantity; }
        public void setQuantity(BigDecimal quantity) { this.quantity = quantity; }
        
        public BigDecimal getCurrentPrice() { return currentPrice; }
        public void setCurrentPrice(BigDecimal currentPrice) { this.currentPrice = currentPrice; }
        
        public BigDecimal getMarketValue() { return marketValue; }
        public void setMarketValue(BigDecimal marketValue) { this.marketValue = marketValue; }
        
        public BigDecimal getCostBasis() { return costBasis; }
        public void setCostBasis(BigDecimal costBasis) { this.costBasis = costBasis; }
        
        public BigDecimal getPnl() { return pnl; }
        public void setPnl(BigDecimal pnl) { this.pnl = pnl; }
        
        public BigDecimal getPnlPercent() { return pnlPercent; }
        public void setPnlPercent(BigDecimal pnlPercent) { this.pnlPercent = pnlPercent; }
        
        public BigDecimal getDayChange() { return dayChange; }
        public void setDayChange(BigDecimal dayChange) { this.dayChange = dayChange; }
        
        public BigDecimal getDayChangePercent() { return dayChangePercent; }
        public void setDayChangePercent(BigDecimal dayChangePercent) { this.dayChangePercent = dayChangePercent; }
        
        public BigDecimal getWeight() { return weight; }
        public void setWeight(BigDecimal weight) { this.weight = weight; }
        
        public LocalDateTime getLastUpdated() { return lastUpdated; }
        public void setLastUpdated(LocalDateTime lastUpdated) { this.lastUpdated = lastUpdated; }
    }
    
    // Getters and Setters
    public Long getPortfolioId() { return portfolioId; }
    public void setPortfolioId(Long portfolioId) { this.portfolioId = portfolioId; }
    
    public String getPortfolioName() { return portfolioName; }
    public void setPortfolioName(String portfolioName) { this.portfolioName = portfolioName; }
    
    public LocalDateTime getTimestamp() { return timestamp; }
    public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }
    
    public BigDecimal getTotalValue() { return totalValue; }
    public void setTotalValue(BigDecimal totalValue) { this.totalValue = totalValue; }
    
    public BigDecimal getCash() { return cash; }
    public void setCash(BigDecimal cash) { this.cash = cash; }
    
    public BigDecimal getTotalCost() { return totalCost; }
    public void setTotalCost(BigDecimal totalCost) { this.totalCost = totalCost; }
    
    public BigDecimal getTotalPnl() { return totalPnl; }
    public void setTotalPnl(BigDecimal totalPnl) { this.totalPnl = totalPnl; }
    
    public BigDecimal getTotalPnlPercent() { return totalPnlPercent; }
    public void setTotalPnlPercent(BigDecimal totalPnlPercent) { this.totalPnlPercent = totalPnlPercent; }
    
    public BigDecimal getDayChange() { return dayChange; }
    public void setDayChange(BigDecimal dayChange) { this.dayChange = dayChange; }
    
    public BigDecimal getDayChangePercent() { return dayChangePercent; }
    public void setDayChangePercent(BigDecimal dayChangePercent) { this.dayChangePercent = dayChangePercent; }
    
    public Map<String, PositionSnapshot> getPositions() { return positions; }
    public void setPositions(Map<String, PositionSnapshot> positions) { this.positions = positions; }
    
    public Map<String, Object> getPerformanceMetrics() { return performanceMetrics; }
    public void setPerformanceMetrics(Map<String, Object> performanceMetrics) { this.performanceMetrics = performanceMetrics; }
}
