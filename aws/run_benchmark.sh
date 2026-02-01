#!/bin/bash
# Run full RAG benchmark
# Usage: nohup bash run_benchmark.sh > benchmark.log 2>&1 &

set -e

# Configuration
OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENAI_API_KEY}}"
export OPENAI_API_KEY

echo "=== RAG Benchmark Runner ==="
echo "Started at: $(date)"
echo ""

# Setup
cd ~/incidentfox
source .venv/bin/activate

# Kill any existing server
pkill -f "ultimate_rag" 2>/dev/null || true
sleep 2

# Start server
echo "Starting Ultimate RAG server..."
nohup python -m ultimate_rag.api.server > ~/server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to start
echo "Waiting for server to initialize (60s)..."
sleep 60

# Check health
if ! curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo "ERROR: Server failed to start!"
    cat ~/server.log | tail -50
    exit 1
fi
echo "Server is healthy!"

# Load persisted tree
echo "Loading persisted tree data..."
LOAD_RESULT=$(curl -s http://localhost:8000/persist/load \
    -H "Content-Type: application/json" \
    -d '{"tree": "default"}')
echo "Load result: $LOAD_RESULT"

# Verify nodes loaded
NODE_COUNT=$(echo "$LOAD_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('node_count', 0))")
echo "Nodes loaded: $NODE_COUNT"

if [ "$NODE_COUNT" -eq 0 ]; then
    echo "ERROR: No nodes loaded! Tree data might be missing."
    exit 1
fi

# Run benchmark
echo ""
echo "=== Starting Full Benchmark ==="
echo "This will take approximately 10-14 hours..."
echo ""

cd ~/rag_benchmarking 2>/dev/null || cd ~

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="full_benchmark_${TIMESTAMP}.log"

python3 scripts/run_multihop_eval.py \
    --api-url http://localhost:8000 \
    --top-k 10 \
    --concurrency 5 \
    --output "multihop_results_${TIMESTAMP}.json" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=== Benchmark Complete ==="
echo "Finished at: $(date)"
echo "Results saved to: $LOG_FILE"
echo ""

# Cleanup
kill $SERVER_PID 2>/dev/null || true

# Show summary
echo "=== Summary ==="
tail -20 "$LOG_FILE"
