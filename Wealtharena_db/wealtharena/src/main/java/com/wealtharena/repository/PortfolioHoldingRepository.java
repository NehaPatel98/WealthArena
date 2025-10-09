package com.wealtharena.repository;

import com.wealtharena.model.Portfolio;
import com.wealtharena.model.PortfolioHolding;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface PortfolioHoldingRepository extends JpaRepository<PortfolioHolding, Long> {
    
    /**
     * Find all holdings for a specific portfolio
     */
    List<PortfolioHolding> findByPortfolio(Portfolio portfolio);
    
    /**
     * Find all holdings for a portfolio by portfolio ID
     */
    List<PortfolioHolding> findByPortfolioId(Long portfolioId);
    
    /**
     * Find a specific holding by portfolio and symbol
     */
    Optional<PortfolioHolding> findByPortfolioAndSymbol(Portfolio portfolio, String symbol);
    
    /**
     * Find a specific holding by portfolio ID and symbol
     */
    Optional<PortfolioHolding> findByPortfolioIdAndSymbol(Long portfolioId, String symbol);
    
    /**
     * Check if a holding exists for a portfolio and symbol
     */
    boolean  existsByPortfolioIdAndSymbol(Long portfolioId, String symbol);
    
    /**
     * Find holdings by symbol across all portfolios
     */
    List<PortfolioHolding> findBySymbol(String symbol); 
    
    /**
     * Find holdings by symbol for a specific user
     */
    @Query("SELECT h FROM PortfolioHolding h JOIN h.portfolio p WHERE h.symbol = :symbol AND p.user.id = :userId")
    List<PortfolioHolding> findBySymbolAndUserId(@Param("symbol") String symbol, @Param("userId") Long userId);
    
    /**
     * Get total quantity of a symbol across all portfolios for a user
     */
    @Query("SELECT COALESCE(SUM(h.quantity), 0) FROM PortfolioHolding h JOIN h.portfolio p WHERE h.symbol = :symbol AND p.user.id = :userId")
    Double getTotalQuantityBySymbolAndUserId(@Param("symbol") String symbol, @Param("userId") Long userId);
    
    /**
     * Get total value of all holdings for a portfolio
     */
    @Query("SELECT COALESCE(SUM(h.totalCost), 0) FROM PortfolioHolding h WHERE h.portfolio.id = :portfolioId")
    Double getTotalValueByPortfolioId(@Param("portfolioId") Long portfolioId);
}
