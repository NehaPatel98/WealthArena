package com.wealtharena.controller;

import com.wealtharena.model.Portfolio;
import com.wealtharena.model.PortfolioHolding;
import com.wealtharena.service.PortfolioService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@RestController
@RequestMapping("/api/portfolios")
public class PortfolioController {
    
    @Autowired
    private PortfolioService portfolioService;
    
    /**
     * GET /api/portfolios - Get all portfolios for a user
     */
    @GetMapping
    public ResponseEntity<List<Portfolio>> getUserPortfolios(@RequestParam Long userId) {
        try {
            List<Portfolio> portfolios = portfolioService.getUserPortfolios(userId);
            return ResponseEntity.ok(portfolios);
        } catch (Exception e) {
            return ResponseEntity.badRequest().build();
        }
    }
    
    /**
     * POST /api/portfolios - Create a new portfolio
     */
    @PostMapping
    public ResponseEntity<?> createPortfolio(@RequestBody CreatePortfolioRequest request) {
        try {
            Portfolio portfolio = portfolioService.createPortfolio(
                request.getUserId(), 
                request.getName(), 
                request.getDescription()
            );
            return ResponseEntity.ok(portfolio);
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("error", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
    
    /**
     * GET /api/portfolios/{id} - Get a specific portfolio
     */
    @GetMapping("/{id}")
    public ResponseEntity<?> getPortfolio(@PathVariable Long id, @RequestParam Long userId) {
        try {
            Optional<Portfolio> portfolio = portfolioService.getPortfolio(id, userId);
            if (portfolio.isPresent()) {
                return ResponseEntity.ok(portfolio.get());
            } else {
                return ResponseEntity.notFound().build();
            }
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("error", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
    
    /**
     * PUT /api/portfolios/{id} - Update portfolio
     */
    @PutMapping("/{id}")
    public ResponseEntity<?> updatePortfolio(@PathVariable Long id, @RequestBody UpdatePortfolioRequest request) {
        try {
            Portfolio portfolio = portfolioService.updatePortfolio(
                id, 
                request.getUserId(), 
                request.getName(), 
                request.getDescription()
            );
            return ResponseEntity.ok(portfolio);
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("error", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
    
    /**
     * DELETE /api/portfolios/{id} - Delete portfolio
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<?> deletePortfolio(@PathVariable Long id, @RequestParam Long userId) {
        try {
            portfolioService.deletePortfolio(id, userId);
            return ResponseEntity.ok().build();
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("error", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
    
    /**
     * POST /api/portfolios/{id}/buy - Buy stocks
     */
    @PostMapping("/{id}/buy")
    public ResponseEntity<?> buyStock(@PathVariable Long id, @RequestBody BuyStockRequest request) {
        try {
            PortfolioHolding holding = portfolioService.buyStock(
                id, 
                request.getUserId(), 
                request.getSymbol(), 
                request.getQuantity(), 
                request.getPrice()
            );
            return ResponseEntity.ok(holding);
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("error", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
    
    /**
     * POST /api/portfolios/{id}/sell - Sell stocks
     */
    @PostMapping("/{id}/sell")
    public ResponseEntity<?> sellStock(@PathVariable Long id, @RequestBody SellStockRequest request) {
        try {
            PortfolioHolding holding = portfolioService.sellStock(
                id, 
                request.getUserId(), 
                request.getSymbol(), 
                request.getQuantity()
            );
            return ResponseEntity.ok(holding);
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("error", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
    
    /**
     * GET /api/portfolios/{id}/holdings - Get portfolio holdings
     */
    @GetMapping("/{id}/holdings")
    public ResponseEntity<?> getPortfolioHoldings(@PathVariable Long id, @RequestParam Long userId) {
        try {
            List<PortfolioHolding> holdings = portfolioService.getPortfolioHoldings(id, userId);
            return ResponseEntity.ok(holdings);
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("error", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
    
    /**
     * GET /api/portfolios/{id}/value - Get portfolio value
     */
    @GetMapping("/{id}/value")
    public ResponseEntity<?> getPortfolioValue(@PathVariable Long id, @RequestParam Long userId) {
        try {
            BigDecimal currentValue = portfolioService.calculatePortfolioValue(id, userId);
            BigDecimal costBasis = portfolioService.calculatePortfolioCostBasis(id, userId);
            BigDecimal pnl = portfolioService.calculatePortfolioPnL(id, userId);
            
            Map<String, Object> valueInfo = new HashMap<>();
            valueInfo.put("currentValue", currentValue);
            valueInfo.put("costBasis", costBasis);
            valueInfo.put("pnl", pnl);
            valueInfo.put("pnlPercentage", costBasis.compareTo(BigDecimal.ZERO) > 0 ? 
                pnl.divide(costBasis, 4, java.math.RoundingMode.HALF_UP).multiply(new BigDecimal("100")) : 
                BigDecimal.ZERO);
            
            return ResponseEntity.ok(valueInfo);
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("error", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
    
    // Request DTOs
    public static class CreatePortfolioRequest {
        private Long userId;
        private String name;
        private String description;
        
        // Getters and Setters
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
    }
    
    public static class UpdatePortfolioRequest {
        private Long userId;
        private String name;
        private String description;
        
        // Getters and Setters
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
    }
    
    public static class BuyStockRequest {
        private Long userId;
        private String symbol;
        private BigDecimal quantity;
        private BigDecimal price;
        
        // Getters and Setters
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getSymbol() { return symbol; }
        public void setSymbol(String symbol) { this.symbol = symbol; }
        public BigDecimal getQuantity() { return quantity; }
        public void setQuantity(BigDecimal quantity) { this.quantity = quantity; }
        public BigDecimal getPrice() { return price; }
        public void setPrice(BigDecimal price) { this.price = price; }
    }
    
    public static class SellStockRequest {
        private Long userId;
        private String symbol;
        private BigDecimal quantity;
        
        // Getters and Setters
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getSymbol() { return symbol; }
        public void setSymbol(String symbol) { this.symbol = symbol; }
        public BigDecimal getQuantity() { return quantity; }
        public void setQuantity(BigDecimal quantity) { this.quantity = quantity; }
    }
}
