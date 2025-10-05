package com.wealtharena.service;

import com.wealtharena.model.SocialSentiment;
import com.wealtharena.repository.SocialSentimentRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;

@Service
public class RedditDataService {
    
    private final SocialSentimentRepository sentimentRepository;
    
    public RedditDataService(SocialSentimentRepository sentimentRepository) {
        this.sentimentRepository = sentimentRepository;
    }
    
    /**
     * Process Reddit data from external source (like your CSV file)
     * This would integrate with Reddit API or process uploaded data
     */
    public void processRedditData(String symbol, List<RedditPost> redditPosts) {
        for (RedditPost post : redditPosts) {
            SocialSentiment sentiment = new SocialSentiment();
            sentiment.setSymbol(symbol);
            sentiment.setPlatform("REDDIT");
            sentiment.setSource(post.getSubreddit());
            sentiment.setContent(post.getTitle() + " " + post.getContent());
            sentiment.setUrl(post.getUrl());
            sentiment.setPostedAt(post.getPostedAt());
            sentiment.setUpvotes(post.getUpvotes());
            sentiment.setComments(post.getComments());
            
            // Simple sentiment analysis (you can integrate with more sophisticated NLP)
            sentiment.setSentimentScore(analyzeSentiment(sentiment.getContent()));
            sentiment.setSentimentLabel(getSentimentLabel(sentiment.getSentimentScore()));
            
            sentimentRepository.save(sentiment);
        }
    }
    
    /**
     * Get recent sentiment for a symbol
     */
    public List<SocialSentiment> getRecentSentiment(String symbol, int days) {
        LocalDateTime since = LocalDateTime.now().minus(days, ChronoUnit.DAYS);
        return sentimentRepository.findBySymbolAndPlatformAndPostedAtAfterOrderByPostedAtDesc(
            symbol, "REDDIT", since);
    }
    
    /**
     * Clean up old data (temporal data as you mentioned)
     */
    public void cleanupOldData(int daysToKeep) {
        LocalDateTime cutoff = LocalDateTime.now().minus(daysToKeep, ChronoUnit.DAYS);
        List<SocialSentiment> oldData = sentimentRepository.findByPostedAtBefore(cutoff);
        sentimentRepository.deleteAll(oldData);
    }
    
    private Double analyzeSentiment(String text) {
        // Simple sentiment analysis - integrate with more sophisticated NLP
        String lowerText = text.toLowerCase();
        int positiveWords = countWords(lowerText, "good", "great", "buy", "bullish", "up", "rise", "profit");
        int negativeWords = countWords(lowerText, "bad", "terrible", "sell", "bearish", "down", "fall", "loss");
        
        if (positiveWords + negativeWords == 0) return 0.0;
        return (double) (positiveWords - negativeWords) / (positiveWords + negativeWords);
    }
    
    private String getSentimentLabel(Double score) {
        if (score > 0.1) return "POSITIVE";
        if (score < -0.1) return "NEGATIVE";
        return "NEUTRAL";
    }
    
    private int countWords(String text, String... words) {
        int count = 0;
        for (String word : words) {
            count += (text.split(word).length - 1);
        }
        return count;
    }
    
    // Inner class for Reddit post structure
    public static class RedditPost {
        private String title;
        private String content;
        private String subreddit;
        private String url;
        private LocalDateTime postedAt;
        private Integer upvotes;
        private Integer comments;
        
        // Getters and setters
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        
        public String getSubreddit() { return subreddit; }
        public void setSubreddit(String subreddit) { this.subreddit = subreddit; }
        
        public String getUrl() { return url; }
        public void setUrl(String url) { this.url = url; }
        
        public LocalDateTime getPostedAt() { return postedAt; }
        public void setPostedAt(LocalDateTime postedAt) { this.postedAt = postedAt; }
        
        public Integer getUpvotes() { return upvotes; }
        public void setUpvotes(Integer upvotes) { this.upvotes = upvotes; }
        
        public Integer getComments() { return comments; }
        public void setComments(Integer comments) { this.comments = comments; }
    }
}

