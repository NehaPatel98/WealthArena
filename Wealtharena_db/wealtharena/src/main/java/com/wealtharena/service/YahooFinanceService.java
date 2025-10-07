package com.wealtharena.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.wealtharena.client.YahooFinanceClient;
import com.wealtharena.model.PriceBar;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.List;

@Service
public class YahooFinanceService {

    private final YahooFinanceClient yahooFinanceClient;

    public YahooFinanceService(YahooFinanceClient yahooFinanceClient) {
        this.yahooFinanceClient = yahooFinanceClient;
    }

    /**
     * Fetches Yahoo daily chart for the given symbol and parses it into PriceBar objects.
     * This method does not persist data; it only returns parsed results.
     */
    public List<PriceBar> fetchDailyPriceBars(String symbol) {
        JsonNode root = yahooFinanceClient.getDailyChart(symbol, "1mo").block();
        List<PriceBar> result = new ArrayList<>();
        if (root == null) return result;

        JsonNode resultArr = root.path("chart").path("result");
        if (!resultArr.isArray() || resultArr.isEmpty()) return result;

        JsonNode first = resultArr.get(0);
        JsonNode timestamps = first.path("timestamp");
        JsonNode indicators = first.path("indicators");
        JsonNode quoteArr = indicators.path("quote");
        JsonNode adjcloseArr = indicators.path("adjclose");
        if (!timestamps.isArray() || !quoteArr.isArray() || quoteArr.isEmpty()) return result;

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

            PriceBar bar = new PriceBar();
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
            result.add(bar);
        }

        return result;
    }

    /**
     * Get the current price for a symbol (latest close price)
     */
    public BigDecimal getCurrentPrice(String symbol) {
        List<PriceBar> bars = fetchDailyPriceBars(symbol);
        if (bars.isEmpty()) {
            throw new RuntimeException("No price data available for symbol: " + symbol);
        }
        // Return the latest close price
        return bars.get(bars.size() - 1).getClose();
    }
    
    /**
     * Fetch only today's data for a symbol
     */
    public List<PriceBar> fetchTodaysData(String symbol) {
        JsonNode root = yahooFinanceClient.getDailyChart(symbol, "1d").block();
        List<PriceBar> result = new ArrayList<>();
        if (root == null) return result;

        JsonNode resultArr = root.path("chart").path("result");
        if (!resultArr.isArray() || resultArr.isEmpty()) return result;

        JsonNode first = resultArr.get(0);
        JsonNode timestamps = first.path("timestamp");
        JsonNode indicators = first.path("indicators");
        JsonNode quoteArr = indicators.path("quote");
        JsonNode adjcloseArr = indicators.path("adjclose");
        if (!timestamps.isArray() || !quoteArr.isArray() || quoteArr.isEmpty()) return result;

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

            PriceBar bar = new PriceBar();
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
            result.add(bar);
        }

        return result;
    }
    
    /**
     * Fetch historical data from 1990 to present for a symbol
     */
    public List<PriceBar> fetchHistoricalData(String symbol) {
        JsonNode root = yahooFinanceClient.getHistoricalData(symbol).block();
        List<PriceBar> result = new ArrayList<>();
        if (root == null) return result;

        JsonNode resultArr = root.path("chart").path("result");
        if (!resultArr.isArray() || resultArr.isEmpty()) return result;

        JsonNode first = resultArr.get(0);
        JsonNode timestamps = first.path("timestamp");
        JsonNode indicators = first.path("indicators");
        JsonNode quoteArr = indicators.path("quote");
        JsonNode adjcloseArr = indicators.path("adjclose");
        if (!timestamps.isArray() || !quoteArr.isArray() || quoteArr.isEmpty()) return result;

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

            PriceBar bar = new PriceBar();
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
            result.add(bar);
        }

        return result;
    }
}


