package com.wealtharena.repository;

import com.wealtharena.model.PriceBar;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;
import org.springframework.data.domain.Pageable;

@Repository
public interface PriceBarRepository extends JpaRepository<PriceBar, Long> {
    Optional<PriceBar> findBySymbolAndTradeDate(String symbol, LocalDate tradeDate);
    List<PriceBar> findBySymbolOrderByTradeDateDesc(String symbol, Pageable pageable);
    Optional<PriceBar> findFirstBySymbolOrderByTradeDateDesc(String symbol);
    boolean existsBySymbolAndTradeDate(String symbol, LocalDate tradeDate);
    List<PriceBar> findByTradeDateBefore(LocalDate tradeDate);
    List<PriceBar> findBySymbolAndTradeDateBetweenOrderByTradeDateAsc(String symbol, LocalDate startDate, LocalDate endDate);
}


