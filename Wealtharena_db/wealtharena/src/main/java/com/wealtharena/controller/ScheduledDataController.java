package com.wealtharena.controller;

import com.wealtharena.service.ScheduledDataService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/scheduled")
public class ScheduledDataController {
    
    @Autowired
    private ScheduledDataService scheduledDataService;
    
    /**
     * Manually trigger daily data fetch for all symbols
     */
    @PostMapping("/fetch-daily")
    public ResponseEntity<?> fetchDailyData() {
        try {
            scheduledDataService.fetchDailyData();
            Map<String, String> response = new HashMap<>();
            response.put("message", "Daily data fetch triggered successfully");
            response.put("status", "success");
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("error", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
    
    /**
     * Manually trigger data fetch for a specific symbol
     */
    @PostMapping("/fetch/{symbol}")
    public ResponseEntity<?> fetchDataForSymbol(@PathVariable String symbol) {
        try {
            scheduledDataService.fetchDataForSymbol(symbol);
            Map<String, String> response = new HashMap<>();
            response.put("message", "Data fetch triggered for symbol: " + symbol);
            response.put("status", "success");
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("error", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
    
    /**
     * Update market metadata
     */
    @PostMapping("/update-metadata")
    public ResponseEntity<?> updateMetadata() {
        try {
            scheduledDataService.updateMarketMetadata();
            Map<String, String> response = new HashMap<>();
            response.put("message", "Market metadata update triggered successfully");
            response.put("status", "success");
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("error", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
}
