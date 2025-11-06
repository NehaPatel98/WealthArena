/**
 * WealthArena Backend Server
 * Main Express Application
 */

import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import routes from './routes';
import { getPool } from './config/db';
import { metricsMiddleware } from './middleware/metrics';
import metricsRouter from './routes/metrics';

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
// Metrics middleware must be applied early to capture all requests
app.use(metricsMiddleware);
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// CORS Configuration - Allow all origins in development for Expo
const isDevelopment = process.env.NODE_ENV !== 'production';

if (isDevelopment) {
  // In development, allow all origins for easier mobile testing
  app.use(cors({
    origin: true,
    credentials: true,
  }));
} else {
  // In production, use strict CORS
  const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || [
    'http://localhost:5001',
    'http://localhost:8081',
    'http://localhost:3000',
  ];

  app.use(cors({
    origin: (origin, callback) => {
      // Allow requests with no origin (mobile apps, Postman, etc.)
      if (!origin) return callback(null, true);
      
      if (allowedOrigins.includes(origin)) {
        callback(null, true);
      } else {
        callback(new Error('Not allowed by CORS'));
      }
    },
    credentials: true,
  }));
}

// Request logging
app.use((req, res, next) => {
  if (process.env.NODE_ENV === 'development') {
    // eslint-disable-next-line no-console
    console.log(`${req.method} ${req.path}`, {
      body: req.body,
      query: req.query,
    });
  }
  next();
});

// Mount metrics endpoint BEFORE generic /api routes to avoid shadowing
app.use('/api/metrics', metricsRouter);
// Mount API routes
app.use('/api', routes);

// Top-level health endpoint (standardized across all services)
app.get('/health', async (req, res) => {
  interface HealthResponse {
    status: string;
    service: string;
    timestamp: string;
    database: string;
    databaseError?: string;
  }

  const health: HealthResponse = {
    status: 'healthy',
    service: 'backend',
    timestamp: new Date().toISOString(),
    database: 'unknown',
  };
  
  // Check database connectivity (non-blocking)
  try {
    const pool = await getPool();
    if (pool && typeof pool.query === 'function') {
      await pool.query('SELECT 1');
      health.database = 'connected';
    } else {
      health.database = 'disconnected';
    }
  } catch (error: unknown) {
    health.database = 'disconnected';
    health.status = 'degraded';
    const errorMessage = error instanceof Error ? error.message : String(error);
    health.databaseError = process.env.NODE_ENV === 'development' ? errorMessage : undefined;
  }
  
  const statusCode = health.status === 'healthy' ? 200 : 503;
  res.status(statusCode).json(health);
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    success: true,
    message: 'Welcome to WealthArena API',
    version: '1.0.0',
    endpoints: {
      health: '/health',
      auth: {
        signup: 'POST /api/auth/signup',
        login: 'POST /api/auth/login',
      },
      signals: {
        top: 'GET /api/signals/top',
        byId: 'GET /api/signals/:signalId',
        bySymbol: 'GET /api/signals/symbol/:symbol',
      },
      portfolio: {
        overview: 'GET /api/portfolio',
        items: 'GET /api/portfolio/items',
        trades: 'GET /api/portfolio/trades',
        positions: 'GET /api/portfolio/positions',
      },
      user: {
        profile: 'GET /api/user/profile',
        xp: 'POST /api/user/xp',
        achievements: 'GET /api/user/achievements',
        quests: 'GET /api/user/quests',
        leaderboard: 'GET /api/user/leaderboard',
      },
      chat: {
        session: 'POST /api/chat/session',
        message: 'POST /api/chat/message',
        history: 'GET /api/chat/history',
        feedback: 'POST /api/chat/feedback',
        sessions: 'GET /api/chat/sessions',
      },
    },
  });
});

// Error handling middleware
app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  // eslint-disable-next-line no-console
  console.error('Error:', err);
  res.status(500).json({
    success: false,
    message: 'Internal server error',
    error: process.env.NODE_ENV === 'development' ? err.message : undefined,
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: 'Endpoint not found',
    path: req.path,
  });
});

// Start server
const startServer = async () => {
  try {
    // Try to connect to database, but don't crash if it fails
    // eslint-disable-next-line no-console
    console.log('Attempting database connection...');
    let dbConnected = false;
    try {
      await getPool();
      dbConnected = true;
      // eslint-disable-next-line no-console
      console.log('Database connection established');
    } catch (dbError: unknown) {
      const errorMessage = dbError instanceof Error ? dbError.message : String(dbError);
      // eslint-disable-next-line no-console
      console.warn('WARNING: Database connection failed, but continuing startup:', errorMessage);
      // eslint-disable-next-line no-console
      console.warn('WARNING: Some features may not work until database is available');
      // eslint-disable-next-line no-console
      console.warn('WARNING: Check database configuration and credentials');
      dbConnected = false;
    }
    
    // Start listening even if database connection failed
    app.listen(PORT, () => {
      // eslint-disable-next-line no-console
      console.log('');
      // eslint-disable-next-line no-console
      console.log('========================================');
      // eslint-disable-next-line no-console
      console.log('WealthArena Backend Server Started');
      // eslint-disable-next-line no-console
      console.log('========================================');
      // eslint-disable-next-line no-console
      console.log(`Port: ${PORT}`);
      // eslint-disable-next-line no-console
      console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
      // eslint-disable-next-line no-console
      console.log(`Database: ${dbConnected ? 'Connected' : 'Disconnected'}`);
      if (process.env.DB_NAME) {
        // eslint-disable-next-line no-console
        console.log(`Database Name: ${process.env.DB_NAME}`);
      }
      if (process.env.DB_HOST) {
        // eslint-disable-next-line no-console
        console.log(`Database Host: ${process.env.DB_HOST}`);
      }
      // eslint-disable-next-line no-console
      console.log('========================================');
      // eslint-disable-next-line no-console
      console.log(`API Documentation: http://localhost:${PORT}/`);
      // eslint-disable-next-line no-console
      console.log(`Health Check: http://localhost:${PORT}/health`);
      // eslint-disable-next-line no-console
      console.log('========================================');
      // eslint-disable-next-line no-console
      console.log('');
    });
  } catch (error) {
    // eslint-disable-next-line no-console
    console.error('ERROR: Failed to start server:', error);
    process.exit(1);
  }
};

// Handle shutdown gracefully
process.on('SIGTERM', () => {
  // eslint-disable-next-line no-console
  console.log('SIGTERM received, shutting down gracefully...');
  process.exit(0);
});

process.on('SIGINT', () => {
  // eslint-disable-next-line no-console
  console.log('\nSIGINT received, shutting down gracefully...');
  process.exit(0);
});

// Start the server
startServer();

export default app;

