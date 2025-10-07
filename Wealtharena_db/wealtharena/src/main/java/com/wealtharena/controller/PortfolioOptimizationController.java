package com.wealtharena.controller;

import com.wealtharena.model.PortfolioOptimizationResult;
import com.wealtharena.service.PortfolioOptimizationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/portfolio/optimization")
@CrossOrigin(origins = "*")
public class PortfolioOptimizationController {

    @Autowired
    private PortfolioOptimizationService optimizationService;

    /**
     * Modern Portfolio Theory Optimization
     * POST /api/portfolio/optimization/mpt
     */
    @PostMapping("/mpt")
    public ResponseEntity<PortfolioOptimizationResult> optimizeMPT(
            @RequestBody MPTRequest request) {
        try {
            PortfolioOptimizationResult result = optimizationService.optimizeModernPortfolioTheory(
                request.getSymbols(), request.getRiskFreeRate());
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            throw new RuntimeException("MPT optimization failed: " + e.getMessage());
        }
    }

    /**
     * Risk Parity Optimization
     * POST /api/portfolio/optimization/risk-parity
     */
    @PostMapping("/risk-parity")
    public ResponseEntity<PortfolioOptimizationResult> optimizeRiskParity(
            @RequestBody RiskParityRequest request) {
        try {
            PortfolioOptimizationResult result = optimizationService.optimizeRiskParity(
                request.getSymbols());
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            throw new RuntimeException("Risk Parity optimization failed: " + e.getMessage());
        }
    }

    /**
     * Black-Litterman Optimization
     * POST /api/portfolio/optimization/black-litterman
     */
    @PostMapping("/black-litterman")
    public ResponseEntity<PortfolioOptimizationResult> optimizeBlackLitterman(
            @RequestBody BlackLittermanRequest request) {
        try {
            PortfolioOptimizationResult result = optimizationService.optimizeBlackLitterman(
                request.getSymbols(), request.getInvestorViews(), 
                request.getRiskFreeRate(), request.getConfidenceLevel());
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            throw new RuntimeException("Black-Litterman optimization failed: " + e.getMessage());
        }
    }

    /**
     * Minimum Variance Optimization
     * POST /api/portfolio/optimization/minimum-variance
     */
    @PostMapping("/minimum-variance")
    public ResponseEntity<PortfolioOptimizationResult> optimizeMinimumVariance(
            @RequestBody MinimumVarianceRequest request) {
        try {
            PortfolioOptimizationResult result = optimizationService.optimizeMinimumVariance(
                request.getSymbols());
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            throw new RuntimeException("Minimum Variance optimization failed: " + e.getMessage());
        }
    }

    /**
     * Get optimization comparison
     * GET /api/portfolio/optimization/compare
     */
    @GetMapping("/compare")
    public ResponseEntity<Map<String, Object>> compareOptimizations(
            @RequestParam List<String> symbols,
            @RequestParam(defaultValue = "0.02") BigDecimal riskFreeRate) {
        try {
            Map<String, Object> comparison = new HashMap<>();
            
            // Run all optimizations
            PortfolioOptimizationResult mpt = optimizationService.optimizeModernPortfolioTheory(symbols, riskFreeRate);
            PortfolioOptimizationResult riskParity = optimizationService.optimizeRiskParity(symbols);
            PortfolioOptimizationResult minVar = optimizationService.optimizeMinimumVariance(symbols);
            
            comparison.put("mpt", mpt);
            comparison.put("riskParity", riskParity);
            comparison.put("minimumVariance", minVar);
            
            // Add summary
            Map<String, Object> summary = new HashMap<>();
            summary.put("bestReturn", findBestReturn(mpt, riskParity, minVar));
            summary.put("lowestRisk", findLowestRisk(mpt, riskParity, minVar));
            summary.put("bestSharpeRatio", findBestSharpeRatio(mpt, riskParity, minVar));
            comparison.put("summary", summary);
            
            return ResponseEntity.ok(comparison);
        } catch (Exception e) {
            throw new RuntimeException("Optimization comparison failed: " + e.getMessage());
        }
    }

    /**
     * Get optimization recommendations
     * GET /api/portfolio/optimization/recommendations
     */
    @GetMapping("/recommendations")
    public ResponseEntity<Map<String, Object>> getRecommendations(
            @RequestParam List<String> symbols,
            @RequestParam(defaultValue = "0.02") BigDecimal riskFreeRate,
            @RequestParam(defaultValue = "moderate") String riskTolerance) {
        try {
            Map<String, Object> recommendations = new HashMap<>();
            
            // Run optimizations
            PortfolioOptimizationResult mpt = optimizationService.optimizeModernPortfolioTheory(symbols, riskFreeRate);
            PortfolioOptimizationResult riskParity = optimizationService.optimizeRiskParity(symbols);
            PortfolioOptimizationResult minVar = optimizationService.optimizeMinimumVariance(symbols);
            
            // Determine best strategy based on risk tolerance
            String recommendedStrategy = determineBestStrategy(riskTolerance, mpt, riskParity, minVar);
            PortfolioOptimizationResult recommendedPortfolio = getRecommendedPortfolio(
                recommendedStrategy, mpt, riskParity, minVar);
            
            recommendations.put("recommendedStrategy", recommendedStrategy);
            recommendations.put("recommendedPortfolio", recommendedPortfolio);
            recommendations.put("riskTolerance", riskTolerance);
            recommendations.put("allOptions", Map.of(
                "mpt", mpt,
                "riskParity", riskParity,
                "minimumVariance", minVar
            ));
            
            return ResponseEntity.ok(recommendations);
        } catch (Exception e) {
            throw new RuntimeException("Recommendations failed: " + e.getMessage());
        }
    }

    // Helper methods
    private String findBestReturn(PortfolioOptimizationResult... results) {
        return Arrays.stream(results)
                .max(Comparator.comparing(PortfolioOptimizationResult::getExpectedReturn))
                .map(PortfolioOptimizationResult::getOptimizationType)
                .orElse("Unknown");
    }

    private String findLowestRisk(PortfolioOptimizationResult... results) {
        return Arrays.stream(results)
                .min(Comparator.comparing(PortfolioOptimizationResult::getExpectedRisk))
                .map(PortfolioOptimizationResult::getOptimizationType)
                .orElse("Unknown");
    }

    private String findBestSharpeRatio(PortfolioOptimizationResult... results) {
        return Arrays.stream(results)
                .max(Comparator.comparing(PortfolioOptimizationResult::getSharpeRatio))
                .map(PortfolioOptimizationResult::getOptimizationType)
                .orElse("Unknown");
    }

    private String determineBestStrategy(String riskTolerance, 
                                      PortfolioOptimizationResult mpt,
                                      PortfolioOptimizationResult riskParity,
                                      PortfolioOptimizationResult minVar) {
        switch (riskTolerance.toLowerCase()) {
            case "conservative":
                return "Minimum Variance";
            case "moderate":
                return "Risk Parity";
            case "aggressive":
                return "Modern Portfolio Theory";
            default:
                return "Risk Parity";
        }
    }

    private PortfolioOptimizationResult getRecommendedPortfolio(String strategy,
                                                              PortfolioOptimizationResult mpt,
                                                              PortfolioOptimizationResult riskParity,
                                                              PortfolioOptimizationResult minVar) {
        switch (strategy) {
            case "Modern Portfolio Theory":
                return mpt;
            case "Risk Parity":
                return riskParity;
            case "Minimum Variance":
                return minVar;
            default:
                return riskParity;
        }
    }

    // Request DTOs
    public static class MPTRequest {
        private List<String> symbols;
        private BigDecimal riskFreeRate = new BigDecimal("0.02");

        public List<String> getSymbols() { return symbols; }
        public void setSymbols(List<String> symbols) { this.symbols = symbols; }
        public BigDecimal getRiskFreeRate() { return riskFreeRate; }
        public void setRiskFreeRate(BigDecimal riskFreeRate) { this.riskFreeRate = riskFreeRate; }
    }

    public static class RiskParityRequest {
        private List<String> symbols;

        public List<String> getSymbols() { return symbols; }
        public void setSymbols(List<String> symbols) { this.symbols = symbols; }
    }

    public static class BlackLittermanRequest {
        private List<String> symbols;
        private Map<String, BigDecimal> investorViews;
        private BigDecimal riskFreeRate = new BigDecimal("0.02");
        private BigDecimal confidenceLevel = new BigDecimal("0.5");

        public List<String> getSymbols() { return symbols; }
        public void setSymbols(List<String> symbols) { this.symbols = symbols; }
        public Map<String, BigDecimal> getInvestorViews() { return investorViews; }
        public void setInvestorViews(Map<String, BigDecimal> investorViews) { this.investorViews = investorViews; }
        public BigDecimal getRiskFreeRate() { return riskFreeRate; }
        public void setRiskFreeRate(BigDecimal riskFreeRate) { this.riskFreeRate = riskFreeRate; }
        public BigDecimal getConfidenceLevel() { return confidenceLevel; }
        public void setConfidenceLevel(BigDecimal confidenceLevel) { this.confidenceLevel = confidenceLevel; }
    }

    public static class MinimumVarianceRequest {
        private List<String> symbols;

        public List<String> getSymbols() { return symbols; }
        public void setSymbols(List<String> symbols) { this.symbols = symbols; }
    }
}
