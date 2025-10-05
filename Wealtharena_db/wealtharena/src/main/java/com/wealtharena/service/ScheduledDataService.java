package com.wealtharena.service;

import com.wealtharena.model.MarketData;
import com.wealtharena.model.PriceBar;
import com.wealtharena.repository.MarketDataRepository;
import com.wealtharena.repository.PriceBarRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.util.List;

@Service
public class ScheduledDataService {
    
    private static final Logger logger = LoggerFactory.getLogger(ScheduledDataService.class);
    
    @Autowired
    private YahooFinanceService yahooFinanceService;
    
    @Autowired
    private MarketDataIngestionService marketDataIngestionService;
    
    @Autowired
    private MarketDataRepository marketDataRepository;
    
    @Autowired
    private PriceBarRepository priceBarRepository;
    
    // List of symbols to fetch daily
    private static final String[] SYMBOLS = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"};
    
    /**
     * Fetch daily data for all symbols at 6:00 PM EST (after market close)
     * This runs every day at 6:00 PM
     */
    @Scheduled(cron = "0 0 18 * * MON-FRI") // 6:00 PM EST, Monday to Friday
    public void fetchDailyData() {
        logger.info("Starting daily data fetch for all symbols...");
        
        for (String symbol : SYMBOLS) {
            try {
                logger.info("Fetching data for symbol: {}", symbol);
                
                // Check if we already have data for today
                LocalDate today = LocalDate.now();
                boolean hasTodayData = priceBarRepository.existsBySymbolAndTradeDate(symbol, today);
                
                if (hasTodayData) {
                    logger.info("Data for {} already exists for today, skipping...", symbol);
                    continue;
                }
                
                // Fetch today's data from Yahoo Finance
                List<PriceBar> priceBars = yahooFinanceService.fetchTodaysData(symbol);
                
                if (!priceBars.isEmpty()) {
                    // Save the data
                    for (PriceBar bar : priceBars) {
                        priceBarRepository.save(bar);
                    }
                    logger.info("Successfully saved {} records for {}", priceBars.size(), symbol);
                } else {
                    logger.warn("No data received for symbol: {}", symbol);
                }
                
                // Small delay between requests to avoid rate limiting
                Thread.sleep(1000);
                
            } catch (Exception e) {
                logger.error("Error fetching data for symbol {}: {}", symbol, e.getMessage());
            }
        }
        
        logger.info("Daily data fetch completed.");
    }
    
    /**
     * Fetch data for a specific symbol (manual trigger)
     */
    public void fetchDataForSymbol(String symbol) {
        try {
            logger.info("Manually fetching data for symbol: {}", symbol);
            
            List<PriceBar> priceBars = yahooFinanceService.fetchDailyPriceBars(symbol);
            
            if (!priceBars.isEmpty()) {
                for (PriceBar bar : priceBars) {
                    priceBarRepository.save(bar);
                }
                logger.info("Successfully saved {} records for {}", priceBars.size(), symbol);
            } else {
                logger.warn("No data received for symbol: {}", symbol);
            }
            
        } catch (Exception e) {
            logger.error("Error fetching data for symbol {}: {}", symbol, e.getMessage());
        }
    }
    
    /**
     * Update market metadata for all symbols
     */
    @Scheduled(cron = "0 0 19 * * MON-FRI") // 7:00 PM EST, Monday to Friday
    public void updateMarketMetadata() {
        logger.info("Updating market metadata...");
        
        for (String symbol : SYMBOLS) {
            try {
                // Check if metadata exists
                if (!marketDataRepository.existsBySymbol(symbol)) {
                    // Create basic metadata entry
                    MarketData marketData = new MarketData();
                    marketData.setSymbol(symbol);
                    marketData.setName(symbol + " Inc."); // Basic name
                    marketData.setExchange("NASDAQ");
                    marketData.setCurrency("USD");
                    marketData.setLastUpdatedAt(java.time.OffsetDateTime.now());
                    
                    marketDataRepository.save(marketData);
                    logger.info("Created metadata for symbol: {}", symbol);
                }
                
            } catch (Exception e) {
                logger.error("Error updating metadata for symbol {}: {}", symbol, e.getMessage());
            }
        }
        
        logger.info("Market metadata update completed.");
    }
    
    /**
     * Clean up old data (keep only last 30 days)
     */
    @Scheduled(cron = "0 0 20 * * SUN") // 8:00 PM EST, Every Sunday
    public void cleanupOldData() {
        logger.info("Starting data cleanup...");
        
        try {
            LocalDate cutoffDate = LocalDate.now().minusDays(30);
            
            // Delete old price bars
            List<PriceBar> oldBars = priceBarRepository.findByTradeDateBefore(cutoffDate);
            if (!oldBars.isEmpty()) {
                priceBarRepository.deleteAll(oldBars);
                logger.info("Deleted {} old price bar records", oldBars.size());
            }
            
        } catch (Exception e) {
            logger.error("Error during data cleanup: {}", e.getMessage());
        }
        
        logger.info("Data cleanup completed.");
    }
}
