package com.wealtharena.controller;

import com.wealtharena.model.BacktestResult;
import com.wealtharena.service.BacktestingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/backtesting")
@CrossOrigin(origins = "*")
public class BacktestingController {

    @Autowired
    private BacktestingService backtestingService;

    /**
     * Backtest Buy and Hold Strategy
     * POST /api/backtesting/buy-and-hold
     */
    @PostMapping("/buy-and-hold")
    public ResponseEntity<BacktestResult> backtestBuyAndHold(@RequestBody BacktestRequest request) {
        try {
            BacktestResult result = backtestingService.backtestBuyAndHold(
                request.getSymbols(), request.getStartDate(), request.getEndDate(), request.getInitialCapital());
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            throw new RuntimeException("Buy and Hold backtest failed: " + e.getMessage());
        }
    }

    /**
     * Backtest Rebalancing Strategy
     * POST /api/backtesting/rebalancing
     */
    @PostMapping("/rebalancing")
    public ResponseEntity<BacktestResult> backtestRebalancing(@RequestBody RebalancingRequest request) {
        try {
            BacktestResult result = backtestingService.backtestRebalancing(
                request.getSymbols(), request.getStartDate(), request.getEndDate(), 
                request.getInitialCapital(), request.getRebalanceFrequencyDays());
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            throw new RuntimeException("Rebalancing backtest failed: " + e.getMessage());
        }
    }

    /**
     * Backtest Momentum Strategy
     * POST /api/backtesting/momentum
     */
    @PostMapping("/momentum")
    public ResponseEntity<BacktestResult> backtestMomentum(@RequestBody MomentumRequest request) {
        try {
            BacktestResult result = backtestingService.backtestMomentum(
                request.getSymbols(), request.getStartDate(), request.getEndDate(), 
                request.getInitialCapital(), request.getLookbackDays(), request.getHoldingDays());
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            throw new RuntimeException("Momentum backtest failed: " + e.getMessage());
        }
    }

    /**
     * Backtest Mean Reversion Strategy
     * POST /api/backtesting/mean-reversion
     */
    @PostMapping("/mean-reversion")
    public ResponseEntity<BacktestResult> backtestMeanReversion(@RequestBody MeanReversionRequest request) {
        try {
            BacktestResult result = backtestingService.backtestMeanReversion(
                request.getSymbols(), request.getStartDate(), request.getEndDate(), 
                request.getInitialCapital(), request.getLookbackDays(), request.getThreshold());
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            throw new RuntimeException("Mean Reversion backtest failed: " + e.getMessage());
        }
    }

    /**
     * Compare Multiple Strategies
     * POST /api/backtesting/compare
     */
    @PostMapping("/compare")
    public ResponseEntity<Map<String, BacktestResult>> compareStrategies(@RequestBody BacktestRequest request) {
        try {
            Map<String, BacktestResult> results = backtestingService.compareStrategies(
                request.getSymbols(), request.getStartDate(), request.getEndDate(), request.getInitialCapital());
            return ResponseEntity.ok(results);
        } catch (Exception e) {
            throw new RuntimeException("Strategy comparison failed: " + e.getMessage());
        }
    }

    /**
     * Get Backtesting Summary
     * GET /api/backtesting/summary
     */
    @GetMapping("/summary")
    public ResponseEntity<Map<String, Object>> getBacktestingSummary(
            @RequestParam List<String> symbols,
            @RequestParam String startDate,
            @RequestParam String endDate,
            @RequestParam(defaultValue = "100000") BigDecimal initialCapital) {
        try {
            LocalDate start = LocalDate.parse(startDate);
            LocalDate end = LocalDate.parse(endDate);
            
            Map<String, BacktestResult> results = backtestingService.compareStrategies(symbols, start, end, initialCapital);
            
            Map<String, Object> summary = new HashMap<>();
            summary.put("strategies", results);
            
            // Find best performing strategy
            String bestStrategy = findBestStrategy(results);
            summary.put("bestStrategy", bestStrategy);
            summary.put("bestResult", results.get(bestStrategy));
            
            // Calculate summary statistics
            Map<String, Object> stats = calculateSummaryStats(results);
            summary.put("summaryStats", stats);
            
            return ResponseEntity.ok(summary);
        } catch (Exception e) {
            throw new RuntimeException("Backtesting summary failed: " + e.getMessage());
        }
    }

    /**
     * Get Strategy Performance Metrics
     * GET /api/backtesting/metrics
     */
    @GetMapping("/metrics")
    public ResponseEntity<Map<String, Object>> getPerformanceMetrics(
            @RequestParam List<String> symbols,
            @RequestParam String startDate,
            @RequestParam String endDate,
            @RequestParam(defaultValue = "100000") BigDecimal initialCapital) {
        try {
            LocalDate start = LocalDate.parse(startDate);
            LocalDate end = LocalDate.parse(endDate);
            
            Map<String, BacktestResult> results = backtestingService.compareStrategies(symbols, start, end, initialCapital);
            
            Map<String, Object> metrics = new HashMap<>();
            
            for (Map.Entry<String, BacktestResult> entry : results.entrySet()) {
                String strategy = entry.getKey();
                BacktestResult result = entry.getValue();
                
                Map<String, Object> strategyMetrics = new HashMap<>();
                strategyMetrics.put("totalReturn", result.getTotalReturn());
                strategyMetrics.put("annualizedReturn", result.getAnnualizedReturn());
                strategyMetrics.put("volatility", result.getVolatility());
                strategyMetrics.put("sharpeRatio", result.getSharpeRatio());
                strategyMetrics.put("maxDrawdown", result.getMaxDrawdown());
                strategyMetrics.put("winRate", result.getWinRate());
                strategyMetrics.put("totalTrades", result.getTotalTrades());
                strategyMetrics.put("profitFactor", result.getProfitFactor());
                
                metrics.put(strategy, strategyMetrics);
            }
            
            return ResponseEntity.ok(metrics);
        } catch (Exception e) {
            throw new RuntimeException("Performance metrics failed: " + e.getMessage());
        }
    }

    // Helper methods
    private String findBestStrategy(Map<String, BacktestResult> results) {
        return results.entrySet().stream()
                .max((entry1, entry2) -> entry1.getValue().getSharpeRatio().compareTo(entry2.getValue().getSharpeRatio()))
                .map(Map.Entry::getKey)
                .orElse("Unknown");
    }

    private Map<String, Object> calculateSummaryStats(Map<String, BacktestResult> results) {
        Map<String, Object> stats = new HashMap<>();
        
        if (results.isEmpty()) {
            return stats;
        }
        
        // Calculate average metrics
        BigDecimal avgReturn = results.values().stream()
                .map(BacktestResult::getTotalReturn)
                .reduce(BigDecimal.ZERO, BigDecimal::add)
                .divide(BigDecimal.valueOf(results.size()), 4, java.math.RoundingMode.HALF_UP);
        
        BigDecimal avgSharpe = results.values().stream()
                .map(BacktestResult::getSharpeRatio)
                .reduce(BigDecimal.ZERO, BigDecimal::add)
                .divide(BigDecimal.valueOf(results.size()), 4, java.math.RoundingMode.HALF_UP);
        
        BigDecimal avgVolatility = results.values().stream()
                .map(BacktestResult::getVolatility)
                .reduce(BigDecimal.ZERO, BigDecimal::add)
                .divide(BigDecimal.valueOf(results.size()), 4, java.math.RoundingMode.HALF_UP);
        
        stats.put("averageReturn", avgReturn);
        stats.put("averageSharpeRatio", avgSharpe);
        stats.put("averageVolatility", avgVolatility);
        stats.put("totalStrategies", results.size());
        
        return stats;
    }

    // Request DTOs
    public static class BacktestRequest {
        private List<String> symbols;
        private LocalDate startDate;
        private LocalDate endDate;
        private BigDecimal initialCapital;

        public List<String> getSymbols() { return symbols; }
        public void setSymbols(List<String> symbols) { this.symbols = symbols; }
        public LocalDate getStartDate() { return startDate; }
        public void setStartDate(LocalDate startDate) { this.startDate = startDate; }
        public LocalDate getEndDate() { return endDate; }
        public void setEndDate(LocalDate endDate) { this.endDate = endDate; }
        public BigDecimal getInitialCapital() { return initialCapital; }
        public void setInitialCapital(BigDecimal initialCapital) { this.initialCapital = initialCapital; }
    }

    public static class RebalancingRequest extends BacktestRequest {
        private int rebalanceFrequencyDays = 30;

        public int getRebalanceFrequencyDays() { return rebalanceFrequencyDays; }
        public void setRebalanceFrequencyDays(int rebalanceFrequencyDays) { this.rebalanceFrequencyDays = rebalanceFrequencyDays; }
    }

    public static class MomentumRequest extends BacktestRequest {
        private int lookbackDays = 20;
        private int holdingDays = 10;

        public int getLookbackDays() { return lookbackDays; }
        public void setLookbackDays(int lookbackDays) { this.lookbackDays = lookbackDays; }
        public int getHoldingDays() { return holdingDays; }
        public void setHoldingDays(int holdingDays) { this.holdingDays = holdingDays; }
    }

    public static class MeanReversionRequest extends BacktestRequest {
        private int lookbackDays = 20;
        private BigDecimal threshold = new BigDecimal("0.02");

        public int getLookbackDays() { return lookbackDays; }
        public void setLookbackDays(int lookbackDays) { this.lookbackDays = lookbackDays; }
        public BigDecimal getThreshold() { return threshold; }
        public void setThreshold(BigDecimal threshold) { this.threshold = threshold; }
    }
}
