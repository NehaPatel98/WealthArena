package com.wealtharena.repository;

import com.wealtharena.model.MarketData;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface MarketDataRepository extends JpaRepository<MarketData, Long> {
    Optional<MarketData> findBySymbol(String symbol);
    boolean existsBySymbol(String symbol);
    List<MarketData> findAllByExchange(String exchange);
}


