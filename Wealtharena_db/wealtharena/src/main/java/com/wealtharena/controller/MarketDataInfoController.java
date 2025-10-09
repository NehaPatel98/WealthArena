package com.wealtharena.controller;

import com.wealtharena.model.MarketData;
import com.wealtharena.repository.MarketDataRepository;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.net.URI;
import java.util.List;

@RestController
@RequestMapping("/api/marketdata/meta")
public class MarketDataInfoController {

    private final MarketDataRepository repository;

    public MarketDataInfoController(MarketDataRepository repository) {
        this.repository = repository;
    }

    @GetMapping
    public List<MarketData> listAll() {
        return repository.findAll();
    }

    @GetMapping("/{symbol}")
    public ResponseEntity<MarketData> getBySymbol(@PathVariable String symbol) {
        return repository.findBySymbol(symbol)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping
    public ResponseEntity<MarketData> upsert(@RequestBody MarketData payload) {
        if (payload.getSymbol() == null || payload.getSymbol().isBlank()) {
            return ResponseEntity.badRequest().build();
        }
        MarketData saved = repository.findBySymbol(payload.getSymbol())
                .map(existing -> {
                    existing.setName(payload.getName());
                    existing.setExchange(payload.getExchange());
                    existing.setCurrency(payload.getCurrency());
                    existing.setSector(payload.getSector());
                    existing.setIndustry(payload.getIndustry());
                    existing.setLastUpdatedAt(payload.getLastUpdatedAt());
                    return repository.save(existing);
                })
                .orElseGet(() -> repository.save(payload));
        return ResponseEntity.created(URI.create("/api/marketdata/meta/" + saved.getSymbol())).body(saved);
    }
}


