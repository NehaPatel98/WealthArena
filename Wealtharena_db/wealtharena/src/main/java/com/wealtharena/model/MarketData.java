package com.wealtharena.model;

import jakarta.persistence.*;
import java.time.OffsetDateTime;

@Entity
@Table(name = "market_data", indexes = {
    @Index(name = "idx_market_data_symbol", columnList = "symbol", unique = true)
})
public class MarketData {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = 32)
    private String symbol;

    @Column(length = 128)
    private String name;

    @Column(length = 64)
    private String exchange;

    @Column(length = 16)
    private String currency;

    @Column(length = 64)
    private String sector;

    @Column(length = 128)
    private String industry;

    @Column
    private OffsetDateTime lastUpdatedAt;

    public Long getId() { return id; }

    public String getSymbol() { return symbol; }

    public void setSymbol(String symbol) { this.symbol = symbol; }

    public String getName() { return name; }

    public void setName(String name) { this.name = name; }

    public String getExchange() { return exchange; }

    public void setExchange(String exchange) { this.exchange = exchange; }

    public String getCurrency() { return currency; }

    public void setCurrency(String currency) { this.currency = currency; }

    public String getSector() { return sector; }

    public void setSector(String sector) { this.sector = sector; }

    public String getIndustry() { return industry; }

    public void setIndustry(String industry) { this.industry = industry; }

    public OffsetDateTime getLastUpdatedAt() { return lastUpdatedAt; }

    public void setLastUpdatedAt(OffsetDateTime lastUpdatedAt) { this.lastUpdatedAt = lastUpdatedAt; }
}


