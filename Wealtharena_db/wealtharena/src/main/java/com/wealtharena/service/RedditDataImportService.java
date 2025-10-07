package com.wealtharena.service;

import com.wealtharena.model.SocialSentiment;
import com.wealtharena.repository.SocialSentimentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

@Service
public class RedditDataImportService {

    @Autowired
    private SocialSentimentRepository socialSentimentRepository;

    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd");
    private static final Pattern TICKER_PATTERN = Pattern.compile("\\b[A-Z]{1,5}\\b");

    public void importRedditData(MultipartFile file) {
        try {
            List<SocialSentiment> sentiments = new ArrayList<>();
            
            BufferedReader reader = new BufferedReader(new InputStreamReader(file.getInputStream()));
            String line;
            boolean isFirstLine = true;
            
            while ((line = reader.readLine()) != null) {
                if (isFirstLine) {
                    isFirstLine = false;
                    continue; // Skip header
                }
                
                String[] columns = line.split(",");
                if (columns.length >= 14) {
                    SocialSentiment sentiment = parseRedditRow(columns);
                    if (sentiment != null) {
                        sentiments.add(sentiment);
                    }
                }
            }
            
            // Save in batches
            socialSentimentRepository.saveAll(sentiments);
            
        } catch (Exception e) {
            throw new RuntimeException("Failed to import Reddit data: " + e.getMessage());
        }
    }

    private SocialSentiment parseRedditRow(String[] columns) {
        try {
            String id = columns[0].trim();
            String subreddit = columns[1].trim();
            String title = columns[2].trim();
            String author = columns[3].trim();
            String contentType = columns[4].trim();
            int score = Integer.parseInt(columns[5].trim());
            int comments = Integer.parseInt(columns[6].trim());
            String createdDate = columns[7].trim();
            String permalink = columns[8].trim();
            String externalUrl = columns[9].trim();
            boolean over18 = Boolean.parseBoolean(columns[10].trim());
            String flair = columns[11].trim();
            String tickers = columns[12].trim();
            String textPreview = columns[13].trim();

            // Extract stock symbols from tickers column
            List<String> symbols = extractStockSymbols(tickers);
            if (symbols.isEmpty()) {
                return null; // Skip if no stock symbols found
            }

            // Calculate sentiment score based on engagement
            double sentimentScore = calculateSentimentScore(score, comments, flair);

            // Create sentiment record for each symbol
            SocialSentiment sentiment = new SocialSentiment();
            sentiment.setSymbol(symbols.get(0)); // Use first symbol
            sentiment.setPlatform("Reddit");
            sentiment.setSource(subreddit);
            sentiment.setContent(title + " " + textPreview);
            sentiment.setUrl(permalink);
            sentiment.setUpvotes(score);
            sentiment.setComments(comments);
            sentiment.setSentimentScore(sentimentScore);
            sentiment.setSentimentLabel(getSentimentLabel(sentimentScore));
            sentiment.setPostedAt(LocalDateTime.now());

            return sentiment;

        } catch (Exception e) {
            return null; // Skip invalid rows
        }
    }

    private List<String> extractStockSymbols(String tickers) {
        List<String> symbols = new ArrayList<>();
        if (tickers != null && !tickers.trim().isEmpty()) {
            String[] tickerArray = tickers.split(",");
            for (String ticker : tickerArray) {
                String cleanTicker = ticker.trim().toUpperCase();
                if (cleanTicker.matches("^[A-Z]{1,5}$")) {
                    symbols.add(cleanTicker);
                }
            }
        }
        return symbols;
    }

    private double calculateSentimentScore(int score, int comments, String flair) {
        // Simple sentiment calculation based on engagement and flair
        double baseScore = 0.5; // Neutral
        
        // Adjust based on upvotes
        if (score > 100) baseScore += 0.3;
        else if (score > 50) baseScore += 0.2;
        else if (score > 10) baseScore += 0.1;
        else if (score < 0) baseScore -= 0.2;
        
        // Adjust based on comments (engagement)
        if (comments > 50) baseScore += 0.2;
        else if (comments > 10) baseScore += 0.1;
        else if (comments > 0) baseScore += 0.05;
        
        // Adjust based on flair
        if (flair != null) {
            String lowerFlair = flair.toLowerCase();
            if (lowerFlair.contains("bull") || lowerFlair.contains("buy")) baseScore += 0.2;
            else if (lowerFlair.contains("bear") || lowerFlair.contains("sell")) baseScore -= 0.2;
            else if (lowerFlair.contains("advice") || lowerFlair.contains("strategy")) baseScore += 0.1;
        }
        
        // Normalize to 0-1 range
        return Math.max(0.0, Math.min(1.0, baseScore));
    }
    
    private String getSentimentLabel(double score) {
        if (score > 0.6) return "POSITIVE";
        if (score < 0.4) return "NEGATIVE";
        return "NEUTRAL";
    }
}
