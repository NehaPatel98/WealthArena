package com.wealtharena.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "social_sentiment")
public class SocialSentiment {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(nullable = false)
    private String symbol;
    
    @Column(nullable = false)
    private String platform; // REDDIT, TWITTER, NEWS
    
    @Column(nullable = false)
    private String source; // subreddit name, twitter handle, news source
    
    @Column(columnDefinition = "TEXT")
    private String content;
    
    @Column
    private String url;
    
    @Column
    private LocalDateTime postedAt;
    
    @Column
    private Double sentimentScore; // -1.0 to 1.0
    
    @Column
    private String sentimentLabel; // POSITIVE, NEGATIVE, NEUTRAL
    
    @Column
    private Integer upvotes;
    
    @Column
    private Integer comments;
    
    @Column
    private LocalDateTime createdAt;
    
    // Constructors
    public SocialSentiment() {
        this.createdAt = LocalDateTime.now();
    }
    
    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    
    public String getSymbol() { return symbol; }
    public void setSymbol(String symbol) { this.symbol = symbol; }
    
    public String getPlatform() { return platform; }
    public void setPlatform(String platform) { this.platform = platform; }
    
    public String getSource() { return source; }
    public void setSource(String source) { this.source = source; }
    
    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }
    
    public String getUrl() { return url; }
    public void setUrl(String url) { this.url = url; }
    
    public LocalDateTime getPostedAt() { return postedAt; }
    public void setPostedAt(LocalDateTime postedAt) { this.postedAt = postedAt; }
    
    public Double getSentimentScore() { return sentimentScore; }
    public void setSentimentScore(Double sentimentScore) { this.sentimentScore = sentimentScore; }
    
    public String getSentimentLabel() { return sentimentLabel; }
    public void setSentimentLabel(String sentimentLabel) { this.sentimentLabel = sentimentLabel; }
    
    public Integer getUpvotes() { return upvotes; }
    public void setUpvotes(Integer upvotes) { this.upvotes = upvotes; }
    
    public Integer getComments() { return comments; }
    public void setComments(Integer comments) { this.comments = comments; }
    
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}
