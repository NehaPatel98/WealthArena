package com.wealtharena.controller;

import com.wealtharena.model.PriceBar;
import com.wealtharena.repository.PriceBarRepository;
import org.springframework.data.domain.PageRequest;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/market")
public class MarketQueryController {

    private final PriceBarRepository priceBarRepository;

    public MarketQueryController(PriceBarRepository priceBarRepository) {
        this.priceBarRepository = priceBarRepository;
    }

    @GetMapping("/data/{symbol}")
    public List<PriceBar> getRecentBars(@PathVariable String symbol, @RequestParam(defaultValue = "10") int limit) {
        int pageSize = Math.max(1, Math.min(limit, 500));
        return priceBarRepository.findBySymbolOrderByTradeDateDesc(symbol, PageRequest.of(0, pageSize));
    }

    @GetMapping("/price/{symbol}")
    public ResponseEntity<PriceBar> getLatestPrice(@PathVariable String symbol) {
        return priceBarRepository.findFirstBySymbolOrderByTradeDateDesc(symbol)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }
}


