/**
 * Database Module Index
 * Re-exports database functions based on DB_TYPE environment variable
 * Supports both SQL Server and PostgreSQL implementations
 */

import dotenv from 'dotenv';

dotenv.config();

// Conditionally import based on DB_TYPE
const dbType = process.env.DB_TYPE || 'sqlserver';

interface DatabaseModule {
  getPool: () => Promise<unknown>;
  executeQuery: <T = unknown>(query: string, params?: Record<string, unknown>) => Promise<unknown>;
  executeProcedure: <T = unknown>(procedureName: string, params?: Record<string, unknown>) => Promise<unknown>;
  closePool: () => Promise<void>;
  executeTransaction?: (queries: Array<{ query: string; params?: unknown[] }>) => Promise<unknown>;
  isGCPDeployment?: () => boolean;
}

let dbModule: DatabaseModule;

if (dbType === 'postgres') {
  dbModule = require('./database-postgres');
} else {
  dbModule = require('./database');
}

// Re-export all database functions
export const getPool = dbModule.getPool;
export const executeQuery = dbModule.executeQuery;
export const executeProcedure = dbModule.executeProcedure;
export const closePool = dbModule.closePool;

// Export optional functions if available
export const executeTransaction = dbModule.executeTransaction;
export const isGCPDeployment = dbModule.isGCPDeployment;

export default dbModule;

