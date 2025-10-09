package com.wealtharena.controller;

import com.wealtharena.service.MarketDataIngestionService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/marketdata")
public class MarketDataController {

    private final MarketDataIngestionService ingestionService;

    public MarketDataController(MarketDataIngestionService ingestionService) {
        this.ingestionService = ingestionService;
    }

    @PostMapping("/ingest/alphaVantage/{symbol}")
    public ResponseEntity<Void> ingestAlpha(@PathVariable String symbol) {
        ingestionService.ingestFromAlphaVantage(symbol);
        return ResponseEntity.accepted().build();
    }

    @PostMapping("/ingest/yahoo/{symbol}")
    public ResponseEntity<Void> ingestYahoo(@PathVariable String symbol) {
        ingestionService.ingestFromYahoo(symbol);
        return ResponseEntity.accepted().build();
    }
    
    @PostMapping("/ingest/historical/{symbol}")
    public ResponseEntity<Void> ingestHistorical(@PathVariable String symbol) {
        ingestionService.ingestHistoricalFromYahoo(symbol);
        return ResponseEntity.accepted().build();
    }
}


