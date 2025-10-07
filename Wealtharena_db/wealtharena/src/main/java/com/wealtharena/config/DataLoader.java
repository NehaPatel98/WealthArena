package com.wealtharena.config;

import com.wealtharena.model.MarketData;
import com.wealtharena.repository.MarketDataRepository;
import com.wealtharena.service.YahooFinanceService;
import com.wealtharena.service.MarketDataIngestionService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import java.time.OffsetDateTime;

@Component
public class DataLoader implements CommandLineRunner {

    private static final Logger log = LoggerFactory.getLogger(DataLoader.class);

    private final MarketDataRepository marketDataRepository;
    private final YahooFinanceService yahooFinanceService;
    private final MarketDataIngestionService ingestionService;

    @Value("${app.data-loader.enabled:true}")
    private boolean enabled;

    @Value("${app.data-loader.symbol:AAPL}")
    private String defaultSymbol;

    public DataLoader(MarketDataRepository marketDataRepository,
                      YahooFinanceService yahooFinanceService,
                      MarketDataIngestionService ingestionService) {
        this.marketDataRepository = marketDataRepository;
        this.yahooFinanceService = yahooFinanceService;
        this.ingestionService = ingestionService;
    }

    @Override
    public void run(String... args) {
        if (!enabled) {
            log.info("DataLoader disabled via property app.data-loader.enabled=false");
            return;
        }

        // Seed basic MarketData if not exists
        marketDataRepository.findBySymbol(defaultSymbol).ifPresentOrElse(existing -> {
            log.info("MarketData already present for {}", defaultSymbol);
        }, () -> {
            MarketData md = new MarketData();
            md.setSymbol(defaultSymbol);
            md.setName("Sample Seed");
            md.setExchange("NASDAQ");
            md.setCurrency("USD");
            md.setSector("Technology");
            md.setIndustry("Consumer Electronics");
            md.setLastUpdatedAt(OffsetDateTime.now());
            marketDataRepository.save(md);
            log.info("Seeded MarketData for {}", defaultSymbol);
        });

        // Try a small Yahoo fetch and persist the data
        try {
            var bars = yahooFinanceService.fetchDailyPriceBars(defaultSymbol);
            log.info("Fetched {} daily bars from Yahoo for {} (not persisted)", bars.size(), defaultSymbol);
            
            // Now actually persist the data using the ingestion service
            ingestionService.ingestFromYahoo(defaultSymbol);
            log.info("Persisted Yahoo data for {}", defaultSymbol);
        } catch (Exception e) {
            log.warn("Yahoo fetch failed during DataLoader: {}", e.getMessage());
        }
    }
}


