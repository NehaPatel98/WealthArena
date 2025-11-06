-- WealthArena Azure SQL Database Schema
-- Run this script to create all necessary tables

-- ==================== Raw Market Data ====================
CREATE TABLE raw_market_data (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    asset_type VARCHAR(20) NOT NULL,  -- 'stock', 'currency_pair', 'commodity', 'crypto'
    date DATE NOT NULL,
    open_price DECIMAL(18, 4) NOT NULL,
    high_price DECIMAL(18, 4) NOT NULL,
    low_price DECIMAL(18, 4) NOT NULL,
    close_price DECIMAL(18, 4) NOT NULL,
    volume BIGINT NOT NULL,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE(),
    
    CONSTRAINT UQ_raw_market_data UNIQUE (symbol, date),
    INDEX IX_raw_market_data_symbol_date (symbol, date),
    INDEX IX_raw_market_data_asset_type (asset_type),
    INDEX IX_raw_market_data_date (date)
);

-- ==================== Processed Features ====================
CREATE TABLE processed_features (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    
    -- Price features
    returns DECIMAL(18, 8),
    log_returns DECIMAL(18, 8),
    volatility_5 DECIMAL(18, 8),
    volatility_20 DECIMAL(18, 8),
    
    -- Technical indicators
    sma_5 DECIMAL(18, 4),
    sma_10 DECIMAL(18, 4),
    sma_20 DECIMAL(18, 4),
    sma_50 DECIMAL(18, 4),
    sma_200 DECIMAL(18, 4),
    ema_12 DECIMAL(18, 4),
    ema_26 DECIMAL(18, 4),
    ema_50 DECIMAL(18, 4),
    rsi DECIMAL(18, 4),
    rsi_6 DECIMAL(18, 4),
    rsi_21 DECIMAL(18, 4),
    macd DECIMAL(18, 8),
    macd_signal DECIMAL(18, 8),
    macd_hist DECIMAL(18, 8),
    bb_upper DECIMAL(18, 4),
    bb_middle DECIMAL(18, 4),
    bb_lower DECIMAL(18, 4),
    bb_width DECIMAL(18, 8),
    bb_position DECIMAL(18, 8),
    atr DECIMAL(18, 4),
    obv BIGINT,
    
    -- Momentum indicators
    momentum_5 DECIMAL(18, 8),
    momentum_10 DECIMAL(18, 8),
    momentum_20 DECIMAL(18, 8),
    
    -- Volume indicators
    volume_sma_20 BIGINT,
    volume_ratio DECIMAL(18, 4),
    
    -- Support/Resistance
    support_20 DECIMAL(18, 4),
    resistance_20 DECIMAL(18, 4),
    price_position DECIMAL(18, 8),
    
    -- Metadata
    processed_at DATETIME2 DEFAULT GETDATE(),
    
    CONSTRAINT UQ_processed_features UNIQUE (symbol, date),
    INDEX IX_processed_features_symbol_date (symbol, date),
    INDEX IX_processed_features_date (date)
);

-- ==================== Model Predictions ====================
CREATE TABLE model_predictions (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    asset_type VARCHAR(20) NOT NULL,
    
    -- Prediction details
    prediction_date DATETIME2 DEFAULT GETDATE(),
    signal VARCHAR(10) NOT NULL,  -- 'BUY', 'SELL', 'HOLD'
    confidence DECIMAL(5, 4) NOT NULL,  -- 0.0000 to 1.0000
    
    -- Entry strategy (from Trading Agent)
    entry_price DECIMAL(18, 4) NOT NULL,
    entry_price_min DECIMAL(18, 4),
    entry_price_max DECIMAL(18, 4),
    entry_timing VARCHAR(20),  -- 'immediate', 'on_pullback', etc.
    
    -- Take Profit levels (from Risk Management Agent)
    tp1_price DECIMAL(18, 4),
    tp1_percent DECIMAL(10, 4),
    tp1_close_percent INT,  -- Percentage of position to close
    tp1_probability DECIMAL(5, 4),
    
    tp2_price DECIMAL(18, 4),
    tp2_percent DECIMAL(10, 4),
    tp2_close_percent INT,
    tp2_probability DECIMAL(5, 4),
    
    tp3_price DECIMAL(18, 4),
    tp3_percent DECIMAL(10, 4),
    tp3_close_percent INT,
    tp3_probability DECIMAL(5, 4),
    
    -- Stop Loss (from Risk Management Agent)
    sl_price DECIMAL(18, 4),
    sl_percent DECIMAL(10, 4),
    sl_type VARCHAR(20),  -- 'fixed', 'trailing'
    sl_trail_amount DECIMAL(18, 4),
    
    -- Risk metrics
    risk_reward_ratio DECIMAL(10, 2),
    max_risk_per_share DECIMAL(18, 4),
    max_reward_per_share DECIMAL(18, 4),
    win_probability DECIMAL(5, 4),
    expected_value DECIMAL(18, 4),
    
    -- Position sizing (from Portfolio Manager Agent)
    recommended_position_percent DECIMAL(10, 4),
    recommended_dollar_amount DECIMAL(18, 2),
    max_risk_percent DECIMAL(10, 4),
    
    -- Model metadata
    model_version VARCHAR(20),
    model_type VARCHAR(50),
    reasoning TEXT,
    
    -- Ranking
    ranking_score DECIMAL(10, 6),
    
    INDEX IX_model_predictions_symbol_date (symbol, prediction_date),
    INDEX IX_model_predictions_asset_type (asset_type),
    INDEX IX_model_predictions_signal (signal),
    INDEX IX_model_predictions_ranking (ranking_score DESC)
);

-- ==================== Model Registry ====================
CREATE TABLE model_registry (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    asset_type VARCHAR(20),
    model_type VARCHAR(50),  -- 'trading_agent', 'risk_management', 'portfolio_manager'
    
    -- Model binary (stored as blob)
    model_blob VARBINARY(MAX),
    
    -- Model metadata
    architecture VARCHAR(50),
    hyperparameters NVARCHAR(MAX),  -- JSON string
    
    -- Performance metrics
    train_sharpe DECIMAL(10, 4),
    train_return DECIMAL(10, 4),
    val_sharpe DECIMAL(10, 4),
    val_return DECIMAL(10, 4),
    backtest_sharpe DECIMAL(10, 4),
    backtest_return DECIMAL(10, 4),
    
    -- Timestamps
    trained_at DATETIME2,
    deployed_at DATETIME2,
    created_at DATETIME2 DEFAULT GETDATE(),
    
    -- Status
    is_active BIT DEFAULT 0,
    is_production BIT DEFAULT 0,
    
    CONSTRAINT UQ_model_version UNIQUE (model_name, version),
    INDEX IX_model_registry_active (is_active, is_production)
);

-- ==================== Portfolio State ====================
CREATE TABLE portfolio_state (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    
    -- Position details
    quantity DECIMAL(18, 8) NOT NULL,
    entry_price DECIMAL(18, 4) NOT NULL,
    entry_date DATE NOT NULL,
    
    -- Current state
    current_price DECIMAL(18, 4),
    current_value DECIMAL(18, 2),
    pnl DECIMAL(18, 2),
    pnl_percent DECIMAL(10, 4),
    
    -- Trade setup (copied from prediction)
    tp1_price DECIMAL(18, 4),
    tp2_price DECIMAL(18, 4),
    tp3_price DECIMAL(18, 4),
    sl_price DECIMAL(18, 4),
    
    -- Timestamps
    updated_at DATETIME2 DEFAULT GETDATE(),
    
    CONSTRAINT UQ_portfolio_user_symbol UNIQUE (user_id, symbol),
    INDEX IX_portfolio_user (user_id),
    INDEX IX_portfolio_updated (updated_at)
);

-- ==================== Game Leaderboard ====================
CREATE TABLE game_leaderboard (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    username VARCHAR(100) NOT NULL,
    
    -- Statistics
    total_score DECIMAL(18, 2) NOT NULL,
    games_played INT DEFAULT 0,
    games_won INT DEFAULT 0,
    
    -- Performance metrics
    avg_return DECIMAL(10, 4),
    avg_sharpe DECIMAL(10, 4),
    avg_max_drawdown DECIMAL(10, 4),
    best_return DECIMAL(10, 4),
    worst_return DECIMAL(10, 4),
    
    -- Timestamps
    last_played DATETIME2,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE(),
    
    CONSTRAINT UQ_leaderboard_user UNIQUE (user_id),
    INDEX IX_leaderboard_score (total_score DESC),
    INDEX IX_leaderboard_username (username)
);

-- ==================== Game Sessions ====================
CREATE TABLE game_sessions (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    game_id VARCHAR(100) NOT NULL UNIQUE,
    user_id VARCHAR(100) NOT NULL,
    
    -- Game parameters
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    initial_capital DECIMAL(18, 2) NOT NULL,
    available_symbols NVARCHAR(MAX),  -- JSON array
    
    -- Game state
    status VARCHAR(20) NOT NULL,  -- 'active', 'completed', 'abandoned'
    current_day DATE,
    
    -- Results
    final_value DECIMAL(18, 2),
    total_return DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    num_trades INT DEFAULT 0,
    
    -- Comparison with agent
    agent_return DECIMAL(10, 4),
    agent_sharpe DECIMAL(10, 4),
    user_vs_agent_score DECIMAL(10, 2),
    
    -- Timestamps
    started_at DATETIME2 DEFAULT GETDATE(),
    completed_at DATETIME2,
    
    INDEX IX_game_sessions_user (user_id),
    INDEX IX_game_sessions_status (status),
    INDEX IX_game_sessions_started (started_at DESC)
);

-- ==================== Audit Log ====================
CREATE TABLE audit_log (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    timestamp DATETIME2 DEFAULT GETDATE(),
    event_type VARCHAR(50) NOT NULL,
    user_id VARCHAR(100),
    symbol VARCHAR(20),
    action VARCHAR(50),
    details NVARCHAR(MAX),  -- JSON
    ip_address VARCHAR(50),
    
    INDEX IX_audit_timestamp (timestamp DESC),
    INDEX IX_audit_user (user_id),
    INDEX IX_audit_event_type (event_type)
);

-- ==================== Create Views for Easy Querying ====================

CREATE VIEW vw_latest_predictions AS
SELECT 
    symbol,
    asset_type,
    signal,
    confidence,
    entry_price,
    tp1_price,
    tp2_price,
    tp3_price,
    sl_price,
    risk_reward_ratio,
    ranking_score,
    model_version,
    prediction_date,
    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY prediction_date DESC) as rn
FROM model_predictions
WHERE prediction_date > DATEADD(day, -1, GETDATE());

CREATE VIEW vw_top_setups_stocks AS
SELECT TOP 10
    symbol,
    signal,
    confidence,
    entry_price,
    tp1_price,
    tp2_price,
    tp3_price,
    sl_price,
    risk_reward_ratio,
    ranking_score
FROM model_predictions
WHERE asset_type = 'stock'
  AND signal IN ('BUY', 'SELL')
  AND prediction_date > DATEADD(day, -1, GETDATE())
ORDER BY ranking_score DESC;

-- ==================== Stored Procedures ====================

CREATE PROCEDURE sp_get_top_setups
    @asset_type VARCHAR(20),
    @count INT = 3
AS
BEGIN
    SELECT TOP (@count)
        symbol,
        asset_type,
        signal,
        confidence,
        entry_price,
        tp1_price,
        tp2_price,
        tp3_price,
        sl_price,
        risk_reward_ratio,
        recommended_position_percent,
        ranking_score,
        model_version,
        reasoning,
        prediction_date
    FROM model_predictions
    WHERE asset_type = @asset_type
      AND signal IN ('BUY', 'SELL')
      AND prediction_date > DATEADD(hour, -24, GETDATE())
    ORDER BY ranking_score DESC;
END;

CREATE PROCEDURE sp_update_portfolio_value
    @user_id VARCHAR(100)
AS
BEGIN
    UPDATE ps
    SET ps.current_value = ps.quantity * ps.current_price,
        ps.pnl = (ps.quantity * ps.current_price) - (ps.quantity * ps.entry_price),
        ps.pnl_percent = ((ps.current_price - ps.entry_price) / ps.entry_price) * 100,
        ps.updated_at = GETDATE()
    FROM portfolio_state ps
    WHERE ps.user_id = @user_id;
END;

-- ==================== Insert Sample Data (for testing) ====================

-- Sample leaderboard entries
INSERT INTO game_leaderboard (user_id, username, total_score, games_played, games_won, avg_return, avg_sharpe)
VALUES 
    ('user_001', 'TradingPro', 2540, 15, 9, 0.254, 2.1),
    ('user_002', 'RL_Master', 2280, 12, 7, 0.228, 1.9),
    ('user_003', 'QuanTitan', 2150, 20, 11, 0.215, 1.8),
    ('user_004', 'MarketGuru', 1920, 8, 5, 0.192, 1.6),
    ('user_005', 'AITrader', 1870, 10, 6, 0.187, 1.5);

-- ==================== Connection String Example ====================
/*
Connection string format for AzureSQL:

Server=tcp:<your-server>.database.windows.net,1433;
Initial Catalog=wealtharena_db;
Persist Security Info=False;
User ID=<username>;
Password=<password>;
MultipleActiveResultSets=False;
Encrypt=True;
TrustServerCertificate=False;
Connection Timeout=30;

Set as environment variable:
export AZURE_SQL_CONNECTION_STRING="mssql+pyodbc://username:password@server/database?driver=ODBC+Driver+17+for+SQL+Server"
*/

