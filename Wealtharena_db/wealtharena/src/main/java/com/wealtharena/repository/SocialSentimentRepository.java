package com.wealtharena.repository;

import com.wealtharena.model.SocialSentiment;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface SocialSentimentRepository extends JpaRepository<SocialSentiment, Long> {
    
    List<SocialSentiment> findBySymbolAndPlatformOrderByPostedAtDesc(String symbol, String platform);
    
    List<SocialSentiment> findBySymbolAndPlatformAndPostedAtAfterOrderByPostedAtDesc(
        String symbol, String platform, LocalDateTime since);
    
    @Query("SELECT s FROM SocialSentiment s WHERE s.symbol = :symbol AND s.postedAt >= :since ORDER BY s.postedAt DESC")
    List<SocialSentiment> findRecentSentimentForSymbol(@Param("symbol") String symbol, @Param("since") LocalDateTime since);
    
    @Query("SELECT s.symbol, AVG(s.sentimentScore) as avgSentiment, COUNT(*) as totalPosts " +
           "FROM SocialSentiment s WHERE s.postedAt >= :since GROUP BY s.symbol")
    List<Object[]> getAverageSentimentBySymbol(@Param("since") LocalDateTime since);
    
    List<SocialSentiment> findByPostedAtBefore(LocalDateTime cutoffDate);
}
