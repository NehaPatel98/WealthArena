/**
 * API Keys Configuration
 * 
 * IMPORTANT: Do NOT commit this file to GitHub!
 * Add this file to .gitignore
 * 
 * For production, use environment variables or a secure key management system
 */

export const API_KEYS = {
  ALPHA_VANTAGE: '5EM9TXSMLJDD83Z8',
  
  // Add other API keys here
  // BACKEND_URL: 'your-backend-url',
  // RL_MODEL_ENDPOINT: 'your-rl-model-endpoint',
};

// Helper function to get API key
export const getAlphaVantageKey = () => {
  return API_KEYS.ALPHA_VANTAGE;
};

// Example usage in components:
// import { getAlphaVantageKey } from '@/config/apiKeys';
// const apiKey = getAlphaVantageKey();

