/**
 * Chatbot Service
 * Handles communication with the WealthArena Chatbot API
 */

const CHATBOT_API_URL = process.env.EXPO_PUBLIC_CHATBOT_API_URL || 
  (__DEV__ ? 'http://localhost:8000' : 'https://wealtharena-chatbot-5224.azurewebsites.net');

export interface ChatMessage {
  message: string;
  user_id?: string;
  context?: string;
}

export interface ChatResponse {
  reply: string;
  tools_used: string[];
  trace_id: string;
}

export interface SentimentAnalysisRequest {
  text: string;
}

export interface SentimentAnalysisResponse {
  text: string;
  sentiment: string;
  confidence: number;
  probabilities: {
    negative: number;
    neutral: number;
    positive: number;
  };
}

export interface PriceRequest {
  ticker: string;
}

export interface PriceResponse {
  ticker: string;
  price: number;
  currency: string;
}

class ChatbotService {
  private readonly baseUrl: string;

  constructor() {
    this.baseUrl = CHATBOT_API_URL;
  }

  /**
   * Send a chat message to the AI assistant
   */
  async chat(message: string, context?: string): Promise<ChatResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          context,
        }),
      });

      if (!response.ok) {
        if (response.status === 404) {
          throw new Error('Chatbot service not available. Please check if the service is running.');
        }
        throw new Error(`Chatbot API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error calling chatbot API:', error);
      
      // Return a fallback response instead of throwing
      return {
        reply: "I'm currently unable to connect to the AI service. Please try again later or contact support if the issue persists.",
        tools_used: [],
        trace_id: `fallback-${Date.now()}`
      };
    }
  }

  /**
   * Analyze sentiment of text
   */
  async analyzeSentiment(text: string): Promise<SentimentAnalysisResponse> {
    try {
      const response = await this.chat(`analyze: ${text}`);
      
      // Parse the sentiment analysis from the response
      // This is a simplified parser - in production, the API should return structured data
      const sentimentRegex = /Predicted Sentiment:\*\* (\w+)/i;
      const confidenceRegex = /Confidence:\*\* ([\d.]+)%/i;
      const sentimentMatch = sentimentRegex.exec(response.reply);
      const confidenceMatch = confidenceRegex.exec(response.reply);
      
      return {
        text,
        sentiment: sentimentMatch ? sentimentMatch[1].toLowerCase() : 'unknown',
        confidence: confidenceMatch ? Number.parseFloat(confidenceMatch[1]) : 0,
        probabilities: {
          negative: 0,
          neutral: 0,
          positive: 0,
        },
      };
    } catch (error) {
      console.error('Error analyzing sentiment:', error);
      throw error;
    }
  }

  /**
   * Get stock price
   */
  async getPrice(ticker: string): Promise<PriceResponse> {
    try {
      const response = await this.chat(`price ${ticker}`);
      
      // Parse the price from the response
      const priceRegex = /\$?([\d,]+\.?\d*)/;
      const priceMatch = priceRegex.exec(response.reply);
      
      return {
        ticker,
        price: priceMatch ? Number.parseFloat(priceMatch[1].replace(',', '')) : 0,
        currency: 'USD',
      };
    } catch (error) {
      console.error('Error getting price:', error);
      throw error;
    }
  }

  /**
   * Get chat history (placeholder)
   */
  async getChatHistory(): Promise<any[]> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/chat/history`, {
        method: 'GET',
      });

      if (!response.ok) {
        throw new Error(`Chatbot API error: ${response.statusText}`);
      }

      const data = await response.json();
      return data.history || [];
    } catch (error) {
      console.error('Error getting chat history:', error);
      throw error;
    }
  }

  /**
   * Clear chat history
   */
  async clearChatHistory(): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/chat/history`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`Chatbot API error: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Error clearing chat history:', error);
      throw error;
    }
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/healthz`);
      return response.ok;
    } catch (error) {
      console.error('Chatbot API health check failed:', error);
      return false;
    }
  }

  /**
   * Start onboarding session
   */
  async startOnboarding(userData: { firstName: string; email: string; userId: number }): Promise<{
    success: boolean;
    sessionId?: string;
    welcomeMessage?: string;
    estimatedQuestions?: number;
    error?: string;
  }> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/onboarding/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          firstName: userData.firstName,
          email: userData.email,
          userId: userData.userId,
        }),
      });

      if (!response.ok) {
        throw new Error(`Onboarding start failed: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return {
        success: true,
        sessionId: data.sessionId,
        welcomeMessage: data.welcomeMessage,
        estimatedQuestions: data.estimatedQuestions,
      };
    } catch (error) {
      console.error('Error starting onboarding:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to start onboarding',
      };
    }
  }

  /**
   * Send onboarding response and get next question
   */
  async sendOnboardingResponse(
    answer: string | string[],
    context: {
      sessionId: string;
      conversationHistory: any[];
      userAnswers: any[];
    }
  ): Promise<{
    success: boolean;
    nextQuestion?: {
      id: string;
      text: string;
      type: 'text' | 'choice' | 'multi' | 'slider' | 'timeline';
      options?: string[];
      mascotVariant?: string;
    };
    estimatedRemaining?: number;
    profileUpdates?: any;
    complete?: boolean;
    error?: string;
  }> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/onboarding/respond`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sessionId: context.sessionId,
          answer,
          conversationHistory: context.conversationHistory,
          userAnswers: context.userAnswers,
        }),
      });

      if (!response.ok) {
        throw new Error(`Onboarding response failed: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return {
        success: true,
        nextQuestion: data.nextQuestion,
        estimatedRemaining: data.estimatedRemaining,
        profileUpdates: data.profileUpdates,
        complete: data.complete,
      };
    } catch (error) {
      console.error('Error sending onboarding response:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to send onboarding response',
      };
    }
  }

  /**
   * Complete onboarding and get final profile
   */
  async completeOnboarding(data: {
    sessionId: string;
    conversationHistory: any[];
    userAnswers: any[];
    userProfile: any;
  }): Promise<{
    success: boolean;
    completionMessage?: string;
    finalProfile?: any;
    rewards?: { xp: number; coins: number };
    error?: string;
  }> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/onboarding/complete`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error(`Onboarding completion failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      return {
        success: true,
        completionMessage: result.completionMessage,
        finalProfile: result.finalProfile,
        rewards: result.rewards,
      };
    } catch (error) {
      console.error('Error completing onboarding:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to complete onboarding',
      };
    }
  }
}

// Export singleton instance
export const chatbotService = new ChatbotService();

