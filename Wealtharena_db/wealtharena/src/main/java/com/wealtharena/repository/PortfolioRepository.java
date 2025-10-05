package com.wealtharena.repository;

import com.wealtharena.model.Portfolio;
import com.wealtharena.model.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface PortfolioRepository extends JpaRepository<Portfolio, Long> {
    
    /**
     * Find all portfolios for a specific user
     */
    List<Portfolio> findByUser(User user);
    
    /**
     * Find all portfolios for a user by user ID
     */
    List<Portfolio> findByUserId(Long userId);
    
    /**
     * Find a specific portfolio by ID and user (for security)
     */
    Optional<Portfolio> findByIdAndUser(Long id, User user);
    
    /**
     * Find a specific portfolio by ID and user ID (for security)
     */
    Optional<Portfolio> findByIdAndUserId(Long id, Long userId);
    
    /**
     * Check if a portfolio exists for a user
     */
    boolean existsByIdAndUserId(Long id, Long userId);
    
    /**
     * Find portfolio by name for a specific user
     */
    Optional<Portfolio> findByNameAndUser(String name, User user);
    
    /**
     * Count portfolios for a user
     */
    long countByUser(User user);
    
    /**
     * Find portfolios with their holdings (eager loading)
     */
    @Query("SELECT p FROM Portfolio p LEFT JOIN FETCH p.holdings WHERE p.user = :user")
    List<Portfolio> findByUserWithHoldings(@Param("user") User user);
    
    /**
     * Find a specific portfolio with its holdings (eager loading)
     */
    @Query("SELECT p FROM Portfolio p LEFT JOIN FETCH p.holdings WHERE p.id = :id AND p.user = :user")
    Optional<Portfolio> findByIdAndUserWithHoldings(@Param("id") Long id, @Param("user") User user);
}
