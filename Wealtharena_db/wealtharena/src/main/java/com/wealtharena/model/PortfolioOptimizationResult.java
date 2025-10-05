package com.wealtharena.model;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

public class PortfolioOptimizationResult {
    
    private String optimizationType;
    private BigDecimal expectedReturn;
    private BigDecimal expectedRisk;
    private BigDecimal sharpeRatio;
    private List<AssetAllocation> allocations;
    private Map<String, Object> metadata;
    private LocalDateTime calculatedAt;
    
    // Constructors
    public PortfolioOptimizationResult() {
        this.calculatedAt = LocalDateTime.now();
    }
    
    public PortfolioOptimizationResult(String optimizationType, BigDecimal expectedReturn, 
                                    BigDecimal expectedRisk, BigDecimal sharpeRatio, 
                                    List<AssetAllocation> allocations) {
        this();
        this.optimizationType = optimizationType;
        this.expectedReturn = expectedReturn;
        this.expectedRisk = expectedRisk;
        this.sharpeRatio = sharpeRatio;
        this.allocations = allocations;
    }
    
    // Inner class for asset allocations
    public static class AssetAllocation {
        private String symbol;
        private BigDecimal weight;
        private BigDecimal expectedReturn;
        private BigDecimal risk;
        private BigDecimal currentPrice;
        private BigDecimal shares;
        
        public AssetAllocation() {}
        
        public AssetAllocation(String symbol, BigDecimal weight, BigDecimal expectedReturn, 
                             BigDecimal risk, BigDecimal currentPrice) {
            this.symbol = symbol;
            this.weight = weight;
            this.expectedReturn = expectedReturn;
            this.risk = risk;
            this.currentPrice = currentPrice;
        }
        
        // Getters and Setters
        public String getSymbol() { return symbol; }
        public void setSymbol(String symbol) { this.symbol = symbol; }
        
        public BigDecimal getWeight() { return weight; }
        public void setWeight(BigDecimal weight) { this.weight = weight; }
        
        public BigDecimal getExpectedReturn() { return expectedReturn; }
        public void setExpectedReturn(BigDecimal expectedReturn) { this.expectedReturn = expectedReturn; }
        
        public BigDecimal getRisk() { return risk; }
        public void setRisk(BigDecimal risk) { this.risk = risk; }
        
        public BigDecimal getCurrentPrice() { return currentPrice; }
        public void setCurrentPrice(BigDecimal currentPrice) { this.currentPrice = currentPrice; }
        
        public BigDecimal getShares() { return shares; }
        public void setShares(BigDecimal shares) { this.shares = shares; }
    }
    
    // Getters and Setters
    public String getOptimizationType() { return optimizationType; }
    public void setOptimizationType(String optimizationType) { this.optimizationType = optimizationType; }
    
    public BigDecimal getExpectedReturn() { return expectedReturn; }
    public void setExpectedReturn(BigDecimal expectedReturn) { this.expectedReturn = expectedReturn; }
    
    public BigDecimal getExpectedRisk() { return expectedRisk; }
    public void setExpectedRisk(BigDecimal expectedRisk) { this.expectedRisk = expectedRisk; }
    
    public BigDecimal getSharpeRatio() { return sharpeRatio; }
    public void setSharpeRatio(BigDecimal sharpeRatio) { this.sharpeRatio = sharpeRatio; }
    
    public List<AssetAllocation> getAllocations() { return allocations; }
    public void setAllocations(List<AssetAllocation> allocations) { this.allocations = allocations; }
    
    public Map<String, Object> getMetadata() { return metadata; }
    public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }
    
    public LocalDateTime getCalculatedAt() { return calculatedAt; }
    public void setCalculatedAt(LocalDateTime calculatedAt) { this.calculatedAt = calculatedAt; }
}
