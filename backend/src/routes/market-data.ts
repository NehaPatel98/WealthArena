/**
 * Market Data Routes
 * Real-time and historical market data endpoints
 */

import express from 'express';
import { executeQuery } from '../config/db';
import { authenticateToken, AuthRequest } from '../middleware/auth';
import { successResponse, errorResponse } from '../utils/responses';

const router = express.Router();

/**
 * GET /api/market-data/symbols
 * Get available trading symbols
 */
router.get('/symbols', authenticateToken, async (req: AuthRequest, res) => {
  try {
    const { asset_type, exchange, limit = 100 } = req.query;

    let query = `
      SELECT DISTINCT Symbol, AssetType, Exchange, Name, Sector, MarketCap
      FROM MarketData
      WHERE 1=1
    `;

    const params: any = {};

    if (asset_type) {
      query += ` AND AssetType = @assetType`;
      params.assetType = asset_type;
    }

    if (exchange) {
      query += ` AND Exchange = @exchange`;
      params.exchange = exchange;
    }

    query += ` ORDER BY MarketCap DESC`;

    if (limit) {
      query = `SELECT TOP (@limit) * FROM (${query}) as ranked`;
      params.limit = parseInt(limit as string);
    }

    const result = await executeQuery(query, params);

    return successResponse(res, result.recordset);
  } catch (error) {
    return errorResponse(res, 'Failed to fetch symbols', 500, error);
  }
});

/**
 * GET /api/market-data/history/:symbol
 * Get historical data for a symbol
 */
router.get('/history/:symbol', authenticateToken, async (req: AuthRequest, res) => {
  try {
    const { symbol } = req.params;
    const { period = '1d', interval = '1m', start_date, end_date } = req.query;

    let query = `
      SELECT 
        Symbol,
        Timestamp,
        Price,
        Volume,
        High,
        Low,
        Open,
        PriceChange1m,
        PriceChange5m,
        PriceChange15m,
        VolumeAvg1h,
        Volatility1h,
        RSI14,
        SMA20,
        SignalStrength
      FROM MarketData
      WHERE Symbol = @symbol
    `;

    const params: any = { symbol };

    // Apply date filters
    if (start_date) {
      query += ` AND Timestamp >= @startDate`;
      params.startDate = start_date;
    }

    if (end_date) {
      query += ` AND Timestamp <= @endDate`;
      params.endDate = end_date;
    }

    // Apply period filter
    if (period === '1d') {
      query += ` AND Timestamp >= DATEADD(day, -1, GETUTCDATE())`;
    } else if (period === '1w') {
      query += ` AND Timestamp >= DATEADD(week, -1, GETUTCDATE())`;
    } else if (period === '1m') {
      query += ` AND Timestamp >= DATEADD(month, -1, GETUTCDATE())`;
    } else if (period === '1y') {
      query += ` AND Timestamp >= DATEADD(year, -1, GETUTCDATE())`;
    }

    query += ` ORDER BY Timestamp DESC`;

    const result = await executeQuery(query, params);

    return successResponse(res, result.recordset);
  } catch (error) {
    return errorResponse(res, 'Failed to fetch historical data', 500, error);
  }
});

/**
 * GET /api/market-data/real-time/:symbol
 * Get real-time data for a symbol
 */
router.get('/real-time/:symbol', authenticateToken, async (req: AuthRequest, res) => {
  try {
    const { symbol } = req.params;

    const query = `
      SELECT TOP 1 
        Symbol,
        Timestamp,
        Price,
        Volume,
        High,
        Low,
        Open,
        PriceChange1m,
        PriceChange5m,
        PriceChange15m,
        VolumeAvg1h,
        Volatility1h,
        RSI14,
        SMA20,
        SignalStrength
      FROM MarketData
      WHERE Symbol = @symbol
      ORDER BY Timestamp DESC
    `;

    const result = await executeQuery(query, { symbol });

    if (result.recordset.length === 0) {
      return errorResponse(res, 'Symbol not found', 404);
    }

    return successResponse(res, result.recordset[0]);
  } catch (error) {
    return errorResponse(res, 'Failed to fetch real-time data', 500, error);
  }
});

/**
 * GET /api/market-data/trending
 * Get trending symbols based on volume and price changes
 */
router.get('/trending', authenticateToken, async (req: AuthRequest, res) => {
  try {
    const { limit = 20, timeframe = '1h' } = req.query;

    let timeFilter = '';
    if (timeframe === '1h') {
      timeFilter = 'AND Timestamp >= DATEADD(hour, -1, GETUTCDATE())';
    } else if (timeframe === '4h') {
      timeFilter = 'AND Timestamp >= DATEADD(hour, -4, GETUTCDATE())';
    } else if (timeframe === '1d') {
      timeFilter = 'AND Timestamp >= DATEADD(day, -1, GETUTCDATE())';
    }

    const query = `
      SELECT TOP (@limit)
        Symbol,
        AssetType,
        AVG(Price) as AvgPrice,
        SUM(Volume) as TotalVolume,
        AVG(PriceChange1m) as AvgPriceChange,
        AVG(Volatility1h) as AvgVolatility,
        COUNT(*) as DataPoints,
        MAX(Timestamp) as LastUpdate
      FROM MarketData
      WHERE 1=1 ${timeFilter}
      GROUP BY Symbol, AssetType
      HAVING COUNT(*) > 10
      ORDER BY TotalVolume DESC, AvgPriceChange DESC
    `;

    const result = await executeQuery(query, { limit: parseInt(limit as string) });

    return successResponse(res, result.recordset);
  } catch (error) {
    return errorResponse(res, 'Failed to fetch trending data', 500, error);
  }
});

/**
 * GET /api/market-data/technical-indicators/:symbol
 * Get technical indicators for a symbol
 */
router.get('/technical-indicators/:symbol', authenticateToken, async (req: AuthRequest, res) => {
  try {
    const { symbol } = req.params;
    const { period = '1d' } = req.query;

    let timeFilter = '';
    if (period === '1d') {
      timeFilter = 'AND Timestamp >= DATEADD(day, -1, GETUTCDATE())';
    } else if (period === '1w') {
      timeFilter = 'AND Timestamp >= DATEADD(week, -1, GETUTCDATE())';
    } else if (period === '1m') {
      timeFilter = 'AND Timestamp >= DATEADD(month, -1, GETUTCDATE())';
    }

    const query = `
      SELECT 
        Symbol,
        Timestamp,
        Price,
        RSI14,
        SMA20,
        SignalStrength,
        PriceChange1m,
        PriceChange5m,
        PriceChange15m,
        Volatility1h,
        VolumeAvg1h
      FROM MarketData
      WHERE Symbol = @symbol ${timeFilter}
      ORDER BY Timestamp DESC
    `;

    const result = await executeQuery(query, { symbol });

    // Calculate additional technical indicators
    const indicators = result.recordset.map((row: any, index: number) => {
      const data = result.recordset.slice(index, index + 20); // Last 20 data points
      
      // Calculate moving averages
      const sma5 = data.slice(0, 5).reduce((sum: number, d: any) => sum + d.Price, 0) / Math.min(5, data.length);
      const sma10 = data.slice(0, 10).reduce((sum: number, d: any) => sum + d.Price, 0) / Math.min(10, data.length);
      
      // Calculate Bollinger Bands (simplified)
      const prices = data.map((d: any) => d.Price);
      const mean = prices.reduce((sum: number, p: number) => sum + p, 0) / prices.length;
      const variance = prices.reduce((sum: number, p: number) => sum + Math.pow(p - mean, 2), 0) / prices.length;
      const stdDev = Math.sqrt(variance);
      
      return {
        ...row,
        SMA5: sma5,
        SMA10: sma10,
        BollingerUpper: mean + (2 * stdDev),
        BollingerLower: mean - (2 * stdDev),
        BollingerMiddle: mean
      };
    });

    return successResponse(res, indicators);
  } catch (error) {
    return errorResponse(res, 'Failed to fetch technical indicators', 500, error);
  }
});

/**
 * GET /api/market-data/screener
 * Advanced market data screener
 */
router.get('/screener', authenticateToken, async (req: AuthRequest, res) => {
  try {
    const {
      asset_type,
      min_price,
      max_price,
      min_volume,
      min_market_cap,
      max_market_cap,
      min_rsi,
      max_rsi,
      signal_strength,
      sort_by = 'market_cap',
      sort_order = 'desc',
      limit = 50
    } = req.query;

    let query = `
      SELECT 
        Symbol,
        AssetType,
        Price,
        Volume,
        MarketCap,
        RSI14,
        SignalStrength,
        PriceChange1m,
        PriceChange5m,
        PriceChange15m,
        Volatility1h,
        MAX(Timestamp) as LastUpdate
      FROM MarketData
      WHERE 1=1
    `;

    const params: any = {};

    if (asset_type) {
      query += ` AND AssetType = @assetType`;
      params.assetType = asset_type;
    }

    if (min_price) {
      query += ` AND Price >= @minPrice`;
      params.minPrice = parseFloat(min_price as string);
    }

    if (max_price) {
      query += ` AND Price <= @maxPrice`;
      params.maxPrice = parseFloat(max_price as string);
    }

    if (min_volume) {
      query += ` AND Volume >= @minVolume`;
      params.minVolume = parseInt(min_volume as string);
    }

    if (min_market_cap) {
      query += ` AND MarketCap >= @minMarketCap`;
      params.minMarketCap = parseInt(min_market_cap as string);
    }

    if (max_market_cap) {
      query += ` AND MarketCap <= @maxMarketCap`;
      params.maxMarketCap = parseInt(max_market_cap as string);
    }

    if (min_rsi) {
      query += ` AND RSI14 >= @minRsi`;
      params.minRsi = parseFloat(min_rsi as string);
    }

    if (max_rsi) {
      query += ` AND RSI14 <= @maxRsi`;
      params.maxRsi = parseFloat(max_rsi as string);
    }

    if (signal_strength) {
      query += ` AND SignalStrength = @signalStrength`;
      params.signalStrength = signal_strength;
    }

    query += ` GROUP BY Symbol, AssetType, Price, Volume, MarketCap, RSI14, SignalStrength, PriceChange1m, PriceChange5m, PriceChange15m, Volatility1h`;

    // Apply sorting
    const validSortFields = ['market_cap', 'price', 'volume', 'rsi', 'price_change'];
    const sortField = validSortFields.includes(sort_by as string) ? sort_by : 'market_cap';
    const sortDirection = sort_order === 'asc' ? 'ASC' : 'DESC';

    if (sortField === 'market_cap') {
      query += ` ORDER BY MarketCap ${sortDirection}`;
    } else if (sortField === 'price') {
      query += ` ORDER BY Price ${sortDirection}`;
    } else if (sortField === 'volume') {
      query += ` ORDER BY Volume ${sortDirection}`;
    } else if (sortField === 'rsi') {
      query += ` ORDER BY RSI14 ${sortDirection}`;
    } else if (sortField === 'price_change') {
      query += ` ORDER BY PriceChange1m ${sortDirection}`;
    }

    query = `SELECT TOP (@limit) * FROM (${query}) as screened`;
    params.limit = parseInt(limit as string);

    const result = await executeQuery(query, params);

    return successResponse(res, result.recordset);
  } catch (error) {
    return errorResponse(res, 'Failed to screen market data', 500, error);
  }
});

export default router;
