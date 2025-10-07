package com.wealtharena.service;

import com.wealtharena.model.Portfolio;
import com.wealtharena.model.PortfolioHolding;
import com.wealtharena.model.User;
import com.wealtharena.repository.PortfolioRepository;
import com.wealtharena.repository.PortfolioHoldingRepository;
import com.wealtharena.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.util.List;
import java.util.Optional;

@Service
@Transactional
public class PortfolioService {
    
    @Autowired
    private PortfolioRepository portfolioRepository;
    
    @Autowired
    private PortfolioHoldingRepository portfolioHoldingRepository;
    
    @Autowired
    private UserRepository userRepository;
    
    @Autowired
    private YahooFinanceService yahooFinanceService;
    
    /**
     * Create a new portfolio for a user
     */
    public Portfolio createPortfolio(Long userId, String name, String description) {
        User user = userRepository.findById(userId)
            .orElseThrow(() -> new RuntimeException("User not found"));
        
        // Check if portfolio name already exists for this user
        if (portfolioRepository.findByNameAndUser(name, user).isPresent()) {
            throw new RuntimeException("Portfolio with name '" + name + "' already exists for this user");
        }
        
        Portfolio portfolio = new Portfolio(name, description, user);
        return portfolioRepository.save(portfolio);
    }
    
    /**
     * Get all portfolios for a user
     */
    @Transactional(readOnly = true)
    public List<Portfolio> getUserPortfolios(Long userId) {
        return portfolioRepository.findByUserId(userId);
    }
    
    /**
     * Get a specific portfolio by ID (with security check)
     */
    @Transactional(readOnly = true)
    public Optional<Portfolio> getPortfolio(Long portfolioId, Long userId) {
        return portfolioRepository.findByIdAndUserId(portfolioId, userId);
    }
    
    /**
     * Update portfolio details
     */
    public Portfolio updatePortfolio(Long portfolioId, Long userId, String name, String description) {
        Portfolio portfolio = portfolioRepository.findByIdAndUserId(portfolioId, userId)
            .orElseThrow(() -> new RuntimeException("Portfolio not found"));
        
        // Check if new name conflicts with existing portfolios
        if (!portfolio.getName().equals(name)) {
            User user = userRepository.findById(userId).orElseThrow();
            if (portfolioRepository.findByNameAndUser(name, user).isPresent()) {
                throw new RuntimeException("Portfolio with name '" + name + "' already exists for this user");
            }
        }
        
        portfolio.setName(name);
        portfolio.setDescription(description);
        return portfolioRepository.save(portfolio);
    }
    
    /**
     * Delete a portfolio
     */
    public void deletePortfolio(Long portfolioId, Long userId) {
        Portfolio portfolio = portfolioRepository.findByIdAndUserId(portfolioId, userId)
            .orElseThrow(() -> new RuntimeException("Portfolio not found"));
        
        portfolioRepository.delete(portfolio);
    }
    
    /**
     * Buy stocks for a portfolio
     */
    public PortfolioHolding buyStock(Long portfolioId, Long userId, String symbol, BigDecimal quantity, BigDecimal price) {
        Portfolio portfolio = portfolioRepository.findByIdAndUserId(portfolioId, userId)
            .orElseThrow(() -> new RuntimeException("Portfolio not found"));
        
        // Check if holding already exists
        Optional<PortfolioHolding> existingHolding = portfolioHoldingRepository
            .findByPortfolioIdAndSymbol(portfolioId, symbol);
        
        if (existingHolding.isPresent()) {
            // Add to existing holding
            PortfolioHolding holding = existingHolding.get();
            holding.addQuantity(quantity, price);
            return portfolioHoldingRepository.save(holding);
        } else {
            // Create new holding
            PortfolioHolding newHolding = new PortfolioHolding(portfolio, symbol, quantity, price);
            return portfolioHoldingRepository.save(newHolding);
        }
    }
    
    /**
     * Sell stocks from a portfolio
     */
    public PortfolioHolding sellStock(Long portfolioId, Long userId, String symbol, BigDecimal quantity) {
        Portfolio portfolio = portfolioRepository.findByIdAndUserId(portfolioId, userId)
            .orElseThrow(() -> new RuntimeException("Portfolio not found"));
        
        PortfolioHolding holding = portfolioHoldingRepository
            .findByPortfolioIdAndSymbol(portfolioId, symbol)
            .orElseThrow(() -> new RuntimeException("No holding found for symbol: " + symbol));
        
        if (holding.getQuantity().compareTo(quantity) < 0) {
            throw new RuntimeException("Insufficient quantity. Available: " + holding.getQuantity() + ", Requested: " + quantity);
        }
        
        holding.subtractQuantity(quantity);
        
        if (holding.getQuantity().compareTo(BigDecimal.ZERO) <= 0) {
            // Remove holding if quantity becomes zero or negative
            portfolioHoldingRepository.delete(holding);
            return null;
        } else {
            return portfolioHoldingRepository.save(holding);
        }
    }
    
    /**
     * Get all holdings for a portfolio
     */
    @Transactional(readOnly = true)
    public List<PortfolioHolding> getPortfolioHoldings(Long portfolioId, Long userId) {
        Portfolio portfolio = portfolioRepository.findByIdAndUserId(portfolioId, userId)
            .orElseThrow(() -> new RuntimeException("Portfolio not found"));
        
        return portfolioHoldingRepository.findByPortfolio(portfolio);
    }
    
    /**
     * Calculate portfolio value (using current market prices)
     */
    @Transactional(readOnly = true)
    public BigDecimal calculatePortfolioValue(Long portfolioId, Long userId) {
        List<PortfolioHolding> holdings = getPortfolioHoldings(portfolioId, userId);
        BigDecimal totalValue = BigDecimal.ZERO;
        
        for (PortfolioHolding holding : holdings) {
            try {
                // Get current price from Yahoo Finance
                BigDecimal currentPrice = yahooFinanceService.getCurrentPrice(holding.getSymbol());
                BigDecimal holdingValue = holding.getQuantity().multiply(currentPrice);
                totalValue = totalValue.add(holdingValue);
            } catch (Exception e) {
                // If we can't get current price, use the cost basis
                totalValue = totalValue.add(holding.getTotalCost());
            }
        }
        
        return totalValue;
    }
    
    /**
     * Calculate portfolio cost basis
     */
    @Transactional(readOnly = true)
    public BigDecimal calculatePortfolioCostBasis(Long portfolioId, Long userId) {
        List<PortfolioHolding> holdings = getPortfolioHoldings(portfolioId, userId);
        return holdings.stream()
            .map(PortfolioHolding::getTotalCost)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
    }
    
    /**
     * Calculate portfolio P&L
     */
    @Transactional(readOnly = true)
    public BigDecimal calculatePortfolioPnL(Long portfolioId, Long userId) {
        BigDecimal currentValue = calculatePortfolioValue(portfolioId, userId);
        BigDecimal costBasis = calculatePortfolioCostBasis(portfolioId, userId);
        return currentValue.subtract(costBasis);
    }
}
