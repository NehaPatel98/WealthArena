package com.wealtharena.client;

import com.fasterxml.jackson.databind.JsonNode;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Component
public class YahooFinanceClient {

    private final WebClient webClient;

    public YahooFinanceClient(@Qualifier("yahooWebClient") WebClient webClient) {
        this.webClient = webClient;
    }

    public Mono<JsonNode> getDailyChart(String symbol) {
        return webClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("/v8/finance/chart/" + symbol)
                        .queryParam("interval", "1d")
                        .queryParam("range", "1mo")  // Last 1 month for daily data
                        .build())
                .retrieve()
                .bodyToMono(JsonNode.class);
    }
    
    public Mono<JsonNode> getDailyChart(String symbol, String range) {
        return webClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("/v8/finance/chart/" + symbol)
                        .queryParam("interval", "1d")
                        .queryParam("range", range)
                        .build())
                .retrieve()
                .bodyToMono(JsonNode.class);
    }
    
    /**
     * Fetch historical data from 1990 to present
     */
    public Mono<JsonNode> getHistoricalData(String symbol) {
        return webClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("/v8/finance/chart/" + symbol)
                        .queryParam("interval", "1d")
                        .queryParam("range", "max")  // Maximum historical data
                        .build())
                .retrieve()
                .bodyToMono(JsonNode.class);
    }
}


