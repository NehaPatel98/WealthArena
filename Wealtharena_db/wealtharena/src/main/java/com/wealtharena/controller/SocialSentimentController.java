package com.wealtharena.controller;

import com.wealtharena.model.SocialSentiment;
import com.wealtharena.service.RedditDataService;
import com.wealtharena.service.RedditDataImportService;
import com.wealtharena.repository.SocialSentimentRepository;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/sentiment")
public class SocialSentimentController {
    
    private final SocialSentimentRepository sentimentRepository;
    private final RedditDataService redditDataService;
    private final RedditDataImportService redditDataImportService;
    
    public SocialSentimentController(SocialSentimentRepository sentimentRepository,
                                   RedditDataService redditDataService,
                                   RedditDataImportService redditDataImportService) {
        this.sentimentRepository = sentimentRepository;
        this.redditDataService = redditDataService;
        this.redditDataImportService = redditDataImportService;
    }
    
    /**
     * Get recent sentiment for a symbol
     */
    @GetMapping("/{symbol}")
    public ResponseEntity<List<SocialSentiment>> getSentimentForSymbol(
            @PathVariable String symbol,
            @RequestParam(defaultValue = "30") int days) {
        
        List<SocialSentiment> sentiment = redditDataService.getRecentSentiment(symbol, days);
        return ResponseEntity.ok(sentiment);
    }
    
    /**
     * Get sentiment summary for a symbol
     */
    @GetMapping("/{symbol}/summary")
    public ResponseEntity<Map<String, Object>> getSentimentSummary(@PathVariable String symbol) {
        LocalDateTime since = LocalDateTime.now().minus(30, ChronoUnit.DAYS);
        List<SocialSentiment> recentSentiment = sentimentRepository.findRecentSentimentForSymbol(symbol, since);
        
        Map<String, Object> summary = new HashMap<>();
        summary.put("symbol", symbol);
        summary.put("totalPosts", recentSentiment.size());
        
        if (!recentSentiment.isEmpty()) {
            double avgSentiment = recentSentiment.stream()
                .mapToDouble(s -> s.getSentimentScore() != null ? s.getSentimentScore() : 0.0)
                .average()
                .orElse(0.0);
            
            long positiveCount = recentSentiment.stream()
                .filter(s -> "POSITIVE".equals(s.getSentimentLabel()))
                .count();
            
            long negativeCount = recentSentiment.stream()
                .filter(s -> "NEGATIVE".equals(s.getSentimentLabel()))
                .count();
            
            summary.put("averageSentiment", avgSentiment);
            summary.put("positivePosts", positiveCount);
            summary.put("negativePosts", negativeCount);
            summary.put("sentimentTrend", avgSentiment > 0.1 ? "BULLISH" : avgSentiment < -0.1 ? "BEARISH" : "NEUTRAL");
        }
        
        return ResponseEntity.ok(summary);
    }
    
    /**
     * Get sentiment for all tracked symbols
     */
    @GetMapping("/summary")
    public ResponseEntity<List<Object[]>> getAllSentimentSummary() {
        LocalDateTime since = LocalDateTime.now().minus(30, ChronoUnit.DAYS);
        List<Object[]> summary = sentimentRepository.getAverageSentimentBySymbol(since);
        return ResponseEntity.ok(summary);
    }
    
    /**
     * Clean up old sentiment data
     */
    @PostMapping("/cleanup")
    public ResponseEntity<Map<String, String>> cleanupOldData(@RequestParam(defaultValue = "90") int daysToKeep) {
        redditDataService.cleanupOldData(daysToKeep);
        Map<String, String> response = new HashMap<>();
        response.put("message", "Cleaned up sentiment data older than " + daysToKeep + " days");
        return ResponseEntity.ok(response);
    }
    
    /**
     * Import Reddit data from CSV file
     */
    @PostMapping("/import/reddit")
    public ResponseEntity<Map<String, String>> importRedditData(@RequestParam("file") MultipartFile file) {
        try {
            if (file.isEmpty()) {
                Map<String, String> error = new HashMap<>();
                error.put("error", "File is empty");
                return ResponseEntity.badRequest().body(error);
            }

            if (!file.getOriginalFilename().toLowerCase().endsWith(".csv")) {
                Map<String, String> error = new HashMap<>();
                error.put("error", "Please upload a CSV file");
                return ResponseEntity.badRequest().body(error);
            }

            redditDataImportService.importRedditData(file);
            Map<String, String> response = new HashMap<>();
            response.put("message", "Reddit data imported successfully!");
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("error", "Import failed: " + e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
    
    /**
     * Get import status
     */
    @GetMapping("/import/status")
    public ResponseEntity<Map<String, String>> getImportStatus() {
        Map<String, String> response = new HashMap<>();
        response.put("status", "Reddit data import service is ready. Upload a CSV file to /api/sentiment/import/reddit");
        return ResponseEntity.ok(response);
    }
}

