
package com.wealtharena.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.wealtharena.client.AlphaVantageClient;
import com.wealtharena.client.YahooFinanceClient;
import com.wealtharena.model.PriceBar;
import com.wealtharena.repository.PriceBarRepository;
import com.wealtharena.service.YahooFinanceService;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.Iterator;

@Service
public class MarketDataIngestionService {

    private final AlphaVantageClient alphaVantageClient;
    private final YahooFinanceClient yahooFinanceClient;
    private final PriceBarRepository priceBarRepository;
    private final YahooFinanceService yahooFinanceService;

    public MarketDataIngestionService(AlphaVantageClient alphaVantageClient,
                                      YahooFinanceClient yahooFinanceClient,
                                      PriceBarRepository priceBarRepository,
                                      YahooFinanceService yahooFinanceService) {
        this.alphaVantageClient = alphaVantageClient;
        this.yahooFinanceClient = yahooFinanceClient;
        this.priceBarRepository = priceBarRepository;
        this.yahooFinanceService = yahooFinanceService;
    }

    @Transactional
    public void ingestFromAlphaVantage(String symbol) {
        JsonNode root = alphaVantageClient.getDailyAdjusted(symbol).block();
        if (root == null) return;
        JsonNode series = root.get("Time Series (Daily)");
        if (series == null || !series.fields().hasNext()) return;
        Iterator<String> dates = series.fieldNames();
        while (dates.hasNext()) {
            String dateStr = dates.next();
            JsonNode day = series.get(dateStr);
            LocalDate date = LocalDate.parse(dateStr);
            PriceBar bar = priceBarRepository.findBySymbolAndTradeDate(symbol, date)
                    .orElseGet(PriceBar::new);
            bar.setSymbol(symbol);
            bar.setTradeDate(date);
            bar.setOpen(new BigDecimal(day.get("1. open").asText()));
            bar.setHigh(new BigDecimal(day.get("2. high").asText()));
            bar.setLow(new BigDecimal(day.get("3. low").asText()));
            bar.setClose(new BigDecimal(day.get("4. close").asText()));
            if (day.has("5. adjusted close")) {
                bar.setAdjustedClose(new BigDecimal(day.get("5. adjusted close").asText()));
            }
            bar.setVolume(day.has("6. volume") ? day.get("6. volume").asLong() : 0L);
            bar.setProvider("AV");
            priceBarRepository.save(bar);
        }
    }

    @Transactional
    public void ingestFromYahoo(String symbol) {
        JsonNode root = yahooFinanceClient.getDailyChart(symbol).block();
        if (root == null) return;
        JsonNode resultArr = root.path("chart").path("result");
        if (!resultArr.isArray() || resultArr.isEmpty()) return;
        JsonNode result = resultArr.get(0);
        JsonNode timestamps = result.path("timestamp");
        JsonNode indicators = result.path("indicators");
        JsonNode quoteArr = indicators.path("quote");
        JsonNode adjcloseArr = indicators.path("adjclose");
        if (!timestamps.isArray() || !quoteArr.isArray() || quoteArr.isEmpty()) return;
        JsonNode quote = quoteArr.get(0);
        JsonNode opens = quote.path("open");
        JsonNode highs = quote.path("high");
        JsonNode lows = quote.path("low");
        JsonNode closes = quote.path("close");
        JsonNode volumes = quote.path("volume");
        JsonNode adjcloses = adjcloseArr.isArray() && !adjcloseArr.isEmpty() ? adjcloseArr.get(0).path("adjclose") : null;

        for (int i = 0; i < timestamps.size(); i++) {
            long ts = timestamps.get(i).asLong();
            LocalDate date = Instant.ofEpochSecond(ts).atZone(ZoneOffset.UTC).toLocalDate();
            if (opens.get(i).isNull() || highs.get(i).isNull() || lows.get(i).isNull() || closes.get(i).isNull()) {
                continue;
            }
            PriceBar bar = priceBarRepository.findBySymbolAndTradeDate(symbol, date)
                    .orElseGet(PriceBar::new);
            bar.setSymbol(symbol);
            bar.setTradeDate(date);
            bar.setOpen(new BigDecimal(opens.get(i).asText()));
            bar.setHigh(new BigDecimal(highs.get(i).asText()));
            bar.setLow(new BigDecimal(lows.get(i).asText()));
            bar.setClose(new BigDecimal(closes.get(i).asText()));
            if (adjcloses != null && i < adjcloses.size() && !adjcloses.get(i).isNull()) {
                bar.setAdjustedClose(new BigDecimal(adjcloses.get(i).asText()));
            }
            bar.setVolume(!volumes.get(i).isNull() ? volumes.get(i).asLong() : 0L);
            bar.setProvider("YF");
            priceBarRepository.save(bar);
        }
    }
    
    @Transactional
    public void ingestHistoricalFromYahoo(String symbol) {
        var bars = yahooFinanceService.fetchHistoricalData(symbol);
        for (PriceBar bar : bars) {
            // Check if record already exists
            priceBarRepository.findBySymbolAndTradeDate(symbol, bar.getTradeDate())
                    .ifPresentOrElse(existing -> {
                        // Update existing record
                        existing.setOpen(bar.getOpen());
                        existing.setHigh(bar.getHigh());
                        existing.setLow(bar.getLow());
                        existing.setClose(bar.getClose());
                        existing.setAdjustedClose(bar.getAdjustedClose());
                        existing.setVolume(bar.getVolume());
                        existing.setProvider(bar.getProvider());
                        priceBarRepository.save(existing);
                    }, () -> {
                        // Save new record
                        priceBarRepository.save(bar);
                    });
        }
    }
}


