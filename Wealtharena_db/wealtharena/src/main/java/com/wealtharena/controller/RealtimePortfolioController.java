package com.wealtharena.controller;

import com.wealtharena.model.PortfolioSnapshot;
import com.wealtharena.service.RealtimePortfolioService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/realtime")
@CrossOrigin(origins = "*")
public class RealtimePortfolioController {

    @Autowired
    private RealtimePortfolioService realtimePortfolioService;

    /**
     * Get real-time portfolio snapshot
     * GET /api/realtime/portfolio/{portfolioId}
     */
    @GetMapping("/portfolio/{portfolioId}")
    public ResponseEntity<PortfolioSnapshot> getPortfolioSnapshot(@PathVariable Long portfolioId) {
        try {
            PortfolioSnapshot snapshot = realtimePortfolioService.getRealtimePortfolioSnapshot(portfolioId);
            return ResponseEntity.ok(snapshot);
        } catch (Exception e) {
            throw new RuntimeException("Failed to get portfolio snapshot: " + e.getMessage());
        }
    }

    /**
     * Get real-time portfolio snapshots for user
     * GET /api/realtime/portfolios/{userId}
     */
    @GetMapping("/portfolios/{userId}")
    public ResponseEntity<List<PortfolioSnapshot>> getUserPortfolioSnapshots(@PathVariable Long userId) {
        try {
            List<PortfolioSnapshot> snapshots = realtimePortfolioService.getRealtimePortfolioSnapshots(userId);
            return ResponseEntity.ok(snapshots);
        } catch (Exception e) {
            throw new RuntimeException("Failed to get user portfolio snapshots: " + e.getMessage());
        }
    }

    /**
     * Subscribe to real-time portfolio updates via SSE
     * GET /api/realtime/portfolio/{portfolioId}/stream
     */
    @GetMapping("/portfolio/{portfolioId}/stream")
    public SseEmitter subscribeToPortfolioUpdates(@PathVariable Long portfolioId) {
        try {
            return realtimePortfolioService.subscribeToPortfolioUpdates(portfolioId);
        } catch (Exception e) {
            throw new RuntimeException("Failed to subscribe to portfolio updates: " + e.getMessage());
        }
    }

    /**
     * Get real-time market prices
     * POST /api/realtime/prices
     */
    @PostMapping("/prices")
    public ResponseEntity<Map<String, java.math.BigDecimal>> getRealtimePrices(@RequestBody PriceRequest request) {
        try {
            Map<String, java.math.BigDecimal> prices = realtimePortfolioService.getRealtimePrices(request.getSymbols());
            return ResponseEntity.ok(prices);
        } catch (Exception e) {
            throw new RuntimeException("Failed to get real-time prices: " + e.getMessage());
        }
    }

    /**
     * Get portfolio performance history
     * GET /api/realtime/portfolio/{portfolioId}/history
     */
    @GetMapping("/portfolio/{portfolioId}/history")
    public ResponseEntity<List<PortfolioSnapshot>> getPortfolioHistory(
            @PathVariable Long portfolioId,
            @RequestParam(defaultValue = "30") int days) {
        try {
            List<PortfolioSnapshot> history = realtimePortfolioService.getPortfolioPerformanceHistory(portfolioId, days);
            return ResponseEntity.ok(history);
        } catch (Exception e) {
            throw new RuntimeException("Failed to get portfolio history: " + e.getMessage());
        }
    }

    /**
     * Get portfolio alerts and notifications
     * GET /api/realtime/portfolio/{portfolioId}/alerts
     */
    @GetMapping("/portfolio/{portfolioId}/alerts")
    public ResponseEntity<List<Map<String, Object>>> getPortfolioAlerts(@PathVariable Long portfolioId) {
        try {
            List<Map<String, Object>> alerts = realtimePortfolioService.getPortfolioAlerts(portfolioId);
            return ResponseEntity.ok(alerts);
        } catch (Exception e) {
            throw new RuntimeException("Failed to get portfolio alerts: " + e.getMessage());
        }
    }

    /**
     * Get real-time portfolio dashboard data
     * GET /api/realtime/dashboard/{userId}
     */
    @GetMapping("/dashboard/{userId}")
    public ResponseEntity<Map<String, Object>> getDashboardData(@PathVariable Long userId) {
        try {
            List<PortfolioSnapshot> snapshots = realtimePortfolioService.getRealtimePortfolioSnapshots(userId);
            
            Map<String, Object> dashboard = new java.util.HashMap<>();
            dashboard.put("portfolios", snapshots);
            dashboard.put("totalValue", snapshots.stream()
                .map(PortfolioSnapshot::getTotalValue)
                .reduce(java.math.BigDecimal.ZERO, java.math.BigDecimal::add));
            dashboard.put("totalPnl", snapshots.stream()
                .map(PortfolioSnapshot::getTotalPnl)
                .reduce(java.math.BigDecimal.ZERO, java.math.BigDecimal::add));
            dashboard.put("portfolioCount", snapshots.size());
            dashboard.put("lastUpdated", java.time.LocalDateTime.now());
            
            return ResponseEntity.ok(dashboard);
        } catch (Exception e) {
            throw new RuntimeException("Failed to get dashboard data: " + e.getMessage());
        }
    }

    /**
     * Get real-time market overview
     * GET /api/realtime/market/overview
     */
    @GetMapping("/market/overview")
    public ResponseEntity<Map<String, Object>> getMarketOverview() {
        try {
            // Get prices for major indices and popular stocks
            List<String> symbols = List.of("AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA");
            Map<String, java.math.BigDecimal> prices = realtimePortfolioService.getRealtimePrices(symbols);
            
            Map<String, Object> overview = new java.util.HashMap<>();
            overview.put("prices", prices);
            overview.put("lastUpdated", java.time.LocalDateTime.now());
            overview.put("marketStatus", "OPEN"); // Simplified - would check actual market hours
            
            return ResponseEntity.ok(overview);
        } catch (Exception e) {
            throw new RuntimeException("Failed to get market overview: " + e.getMessage());
        }
    }

    /**
     * Get real-time portfolio comparison
     * GET /api/realtime/compare/{userId}
     */
    @GetMapping("/compare/{userId}")
    public ResponseEntity<Map<String, Object>> comparePortfolios(@PathVariable Long userId) {
        try {
            List<PortfolioSnapshot> snapshots = realtimePortfolioService.getRealtimePortfolioSnapshots(userId);
            
            Map<String, Object> comparison = new java.util.HashMap<>();
            comparison.put("portfolios", snapshots);
            
            // Find best and worst performing portfolios
            Optional<PortfolioSnapshot> bestPortfolio = snapshots.stream()
                .max(Comparator.comparing(PortfolioSnapshot::getTotalPnlPercent));
            Optional<PortfolioSnapshot> worstPortfolio = snapshots.stream()
                .min(Comparator.comparing(PortfolioSnapshot::getTotalPnlPercent));
            
            if (bestPortfolio.isPresent()) {
                comparison.put("bestPerformer", bestPortfolio.get().getPortfolioName());
                comparison.put("bestReturn", bestPortfolio.get().getTotalPnlPercent());
            }
            
            if (worstPortfolio.isPresent()) {
                comparison.put("worstPerformer", worstPortfolio.get().getPortfolioName());
                comparison.put("worstReturn", worstPortfolio.get().getTotalPnlPercent());
            }
            
            // Calculate average performance
            double avgReturn = snapshots.stream()
                .mapToDouble(s -> s.getTotalPnlPercent().doubleValue())
                .average()
                .orElse(0.0);
            comparison.put("averageReturn", avgReturn);
            
            return ResponseEntity.ok(comparison);
        } catch (Exception e) {
            throw new RuntimeException("Failed to compare portfolios: " + e.getMessage());
        }
    }

    /**
     * Get real-time portfolio analytics
     * GET /api/realtime/portfolio/{portfolioId}/analytics
     */
    @GetMapping("/portfolio/{portfolioId}/analytics")
    public ResponseEntity<Map<String, Object>> getPortfolioAnalytics(@PathVariable Long portfolioId) {
        try {
            PortfolioSnapshot snapshot = realtimePortfolioService.getRealtimePortfolioSnapshot(portfolioId);
            
            Map<String, Object> analytics = new java.util.HashMap<>();
            analytics.put("snapshot", snapshot);
            analytics.put("performanceMetrics", snapshot.getPerformanceMetrics());
            analytics.put("alerts", realtimePortfolioService.getPortfolioAlerts(portfolioId));
            analytics.put("lastUpdated", java.time.LocalDateTime.now());
            
            return ResponseEntity.ok(analytics);
        } catch (Exception e) {
            throw new RuntimeException("Failed to get portfolio analytics: " + e.getMessage());
        }
    }

    // Request DTOs
    public static class PriceRequest {
        private List<String> symbols;

        public List<String> getSymbols() { return symbols; }
        public void setSymbols(List<String> symbols) { this.symbols = symbols; }
    }
}
