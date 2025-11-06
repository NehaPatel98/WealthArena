#!/bin/bash
# WealthArena Demo Startup Script
# This script starts all services needed for the demo

echo "🚀 Starting WealthArena Demo..."
echo ""

# Set project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Colors for output
GREEN='\033[0.32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create necessary directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p "$PROJECT_ROOT/data/raw"
mkdir -p "$PROJECT_ROOT/data/processed"
mkdir -p "$PROJECT_ROOT/data/features"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/airflow/dags"

# Check if virtual environment exists
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Creating...${NC}"
    python3 -m venv "$PROJECT_ROOT/venv"
fi

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source "$PROJECT_ROOT/venv/bin/activate"

# Install dependencies
echo -e "${BLUE}📦 Installing dependencies...${NC}"
pip install -q -r "$PROJECT_ROOT/requirements_demo.txt"

# Initialize Airflow (if not already done)
export AIRFLOW_HOME="$PROJECT_ROOT/airflow"
if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
    echo -e "${BLUE}🔄 Initializing Airflow...${NC}"
    airflow db init
    
    # Create admin user
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@wealtharena.com \
        --password admin
fi

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Starting services..."
echo ""

# Function to kill all background processes on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Stopping all services...${NC}"
    kill $(jobs -p) 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# Start Airflow Webserver
echo -e "${BLUE}🌐 Starting Airflow Webserver on http://localhost:8080${NC}"
airflow webserver --port 8080 > "$PROJECT_ROOT/logs/airflow_webserver.log" 2>&1 &

# Wait a bit for webserver to start
sleep 3

# Start Airflow Scheduler
echo -e "${BLUE}📅 Starting Airflow Scheduler${NC}"
airflow scheduler > "$PROJECT_ROOT/logs/airflow_scheduler.log" 2>&1 &

# Wait for scheduler to start
sleep 2

# Start Backend API
echo -e "${BLUE}🔌 Starting Backend API on http://localhost:8000${NC}"
cd "$PROJECT_ROOT"
python backend/main.py > "$PROJECT_ROOT/logs/backend.log" 2>&1 &

# Wait for backend to start
sleep 2

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}🎉 WealthArena Demo is Running!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "📊 Services:"
echo -e "  • Airflow UI:    ${BLUE}http://localhost:8080${NC} (admin/admin)"
echo -e "  • Backend API:   ${BLUE}http://localhost:8000${NC}"
echo -e "  • API Docs:      ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo -e "📝 Logs:"
echo -e "  • Airflow Web:   tail -f logs/airflow_webserver.log"
echo -e "  • Airflow Sched: tail -f logs/airflow_scheduler.log"
echo -e "  • Backend:       tail -f logs/backend.log"
echo ""
echo -e "🎮 Next Steps:"
echo -e "  1. Open Airflow UI and trigger 'fetch_market_data' DAG"
echo -e "  2. After data fetch, trigger 'preprocess_market_data' DAG"
echo -e "  3. Check API at http://localhost:8000/docs"
echo -e "  4. Start the frontend (see DEMO_SETUP_GUIDE.md)"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for user interrupt
wait

