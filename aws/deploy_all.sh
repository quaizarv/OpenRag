#!/bin/bash
# One-click deploy and run benchmark on AWS
# Usage: ./deploy_all.sh
#
# This script deploys a pre-built RAPTOR tree (~70% Recall@10 with Cohere reranker)
# and runs the full 2556-query MultiHop-RAG evaluation (~8 hours)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=== One-Click AWS Benchmark Deployment ==="
echo ""
echo "Tree stats:"
echo "  - 1137 nodes with RAPTOR hierarchy (4 layers)"
echo "  - 3072-dim embeddings (text-embedding-3-large)"
echo "  - 71.67% Recall@10 verified on 30 queries"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v aws &> /dev/null; then
    echo "ERROR: AWS CLI not installed"
    echo "Install with: brew install awscli"
    echo "Then run: aws configure"
    exit 1
fi

if ! aws sts get-caller-identity &> /dev/null; then
    echo "ERROR: AWS credentials not configured"
    echo "Run: aws configure"
    exit 1
fi

# Check for key pair
if [ -z "$AWS_KEY_NAME" ]; then
    echo ""
    echo "Available key pairs:"
    aws ec2 describe-key-pairs --query 'KeyPairs[*].KeyName' --output table
    echo ""
    read -p "Enter key pair name: " AWS_KEY_NAME
    export AWS_KEY_NAME
fi

KEY_FILE="$HOME/.ssh/${AWS_KEY_NAME}.pem"
if [ ! -f "$KEY_FILE" ]; then
    echo "ERROR: Key file not found: $KEY_FILE"
    exit 1
fi

# Check for tree data
TREES_DIR="/Users/apple/Desktop/incidentfox/trees"
if [ ! -f "$TREES_DIR/default.pkl" ]; then
    echo "ERROR: Tree data not found at $TREES_DIR/default.pkl"
    echo "Make sure the tree is saved from the local server first!"
    exit 1
fi

# Verify tree has nodes
echo "Verifying tree integrity..."
python3 -c "
import pickle
with open('$TREES_DIR/default.pkl', 'rb') as f:
    d = pickle.load(f)
nodes = d.get('nodes', {})
layers = d.get('layer_to_node_indices', {})
if len(nodes) < 1000:
    print(f'WARNING: Tree has only {len(nodes)} nodes, expected 1137')
    exit(1)
print(f'✓ Tree verified: {len(nodes)} nodes, {len(layers)} layers')
"

echo "✓ AWS CLI configured"
echo "✓ Key pair: $AWS_KEY_NAME"
echo "✓ Tree data verified (32MB)"
echo ""

# Check for OpenAI key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Enter your OpenAI API key (for embeddings):"
    read -s OPENAI_API_KEY
    export OPENAI_API_KEY
fi
echo "✓ OpenAI API key set"

# Check for Cohere key (for SOTA reranking)
if [ -z "$COHERE_API_KEY" ]; then
    echo "Enter your Cohere API key (for SOTA reranking, or press Enter to skip):"
    read -s COHERE_API_KEY
    export COHERE_API_KEY
fi
if [ -n "$COHERE_API_KEY" ]; then
    echo "✓ Cohere API key set (SOTA reranking enabled)"
else
    echo "  Cohere key not set, will use cross-encoder fallback"
fi
echo ""

# Launch EC2
echo "Step 1: Launching EC2 instance (g4dn.xlarge, 4 vCPU, 16GB RAM)..."
bash aws/launch_ec2.sh

# Get instance info
INSTANCE_ID=$(cat /tmp/rag_benchmark_instance.txt)
PUBLIC_IP=$(cat /tmp/rag_benchmark_ip.txt)

echo ""
echo "Step 2: Waiting for instance to be ready (2 minutes)..."
sleep 120

# SSH test
echo "Testing SSH connection..."
for i in {1..5}; do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$KEY_FILE" "ubuntu@${PUBLIC_IP}" "echo 'SSH OK'" 2>/dev/null; then
        break
    fi
    echo "  Retry $i..."
    sleep 30
done

# Upload tree data
echo ""
echo "Step 3: Uploading tree data (~32MB)..."
scp -o StrictHostKeyChecking=no -i "$KEY_FILE" "$TREES_DIR/default.pkl" "ubuntu@${PUBLIC_IP}:~/"

# Upload benchmark files
echo ""
echo "Step 4: Uploading benchmark scripts and corpus..."
scp -o StrictHostKeyChecking=no -i "$KEY_FILE" \
    aws/run_benchmark.sh \
    "ubuntu@${PUBLIC_IP}:~/"

# Create rag_benchmarking directory structure and upload
ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" "ubuntu@${PUBLIC_IP}" "mkdir -p ~/rag_benchmarking/adapters ~/rag_benchmarking/scripts ~/rag_benchmarking/multihop_rag/dataset"

scp -o StrictHostKeyChecking=no -i "$KEY_FILE" \
    adapters/ultimate_rag_adapter.py \
    "ubuntu@${PUBLIC_IP}:~/rag_benchmarking/adapters/"

scp -o StrictHostKeyChecking=no -i "$KEY_FILE" \
    scripts/run_multihop_eval.py \
    "ubuntu@${PUBLIC_IP}:~/rag_benchmarking/scripts/"

# Upload corpus (large file)
echo "Uploading corpus (~130MB)..."
scp -o StrictHostKeyChecking=no -i "$KEY_FILE" \
    multihop_rag/dataset/corpus.json \
    multihop_rag/dataset/MultiHopRAG.json \
    "ubuntu@${PUBLIC_IP}:~/rag_benchmarking/multihop_rag/dataset/"

# Run setup
echo ""
echo "Step 5: Setting up environment on EC2..."
ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" "ubuntu@${PUBLIC_IP}" << 'REMOTE_SETUP'
set -e

# Install system dependencies
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git

# Clone incidentfox
cd ~
if [ ! -d "incidentfox" ]; then
    git clone https://github.com/incidentfox/incidentfox.git
fi

cd ~/incidentfox
git fetch origin
git checkout feat/raptor-hierarchy-and-graph-integration
git reset --hard origin/feat/raptor-hierarchy-and-graph-integration

# Setup venv and install dependencies
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install additional dependencies for cross-encoder, RAPTOR, and Cohere
pip install sentence-transformers umap-learn requests tqdm tenacity cohere

# Install incidentfox as editable package
pip install -e .

# Move tree data to correct location
mkdir -p ~/incidentfox/trees
mv ~/default.pkl ~/incidentfox/trees/default.pkl 2>/dev/null || true

echo ""
echo "=== Setup Complete ==="
echo "Tree location: ~/incidentfox/trees/default.pkl"
ls -lh ~/incidentfox/trees/
REMOTE_SETUP

echo ""
echo "Step 6: Starting the benchmark (in background)..."

# Create and run the benchmark script on EC2
ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" "ubuntu@${PUBLIC_IP}" << REMOTE_RUN
set -e
cd ~/incidentfox
source .venv/bin/activate

export OPENAI_API_KEY="$OPENAI_API_KEY"
export COHERE_API_KEY="$COHERE_API_KEY"
export PYTHONPATH="\$HOME/incidentfox:\$HOME/incidentfox/knowledge_base:\$PYTHONPATH"

# Start server in background
echo "Starting Ultimate RAG server..."
nohup python -m ultimate_rag.api.server > ~/server.log 2>&1 &
SERVER_PID=\$!
echo "Server PID: \$SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to start..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    sleep 5
    echo "  Waiting... (\$i/60)"
done

# Load the pre-built tree
echo "Loading RAPTOR tree..."
curl -s -X POST "http://localhost:8000/persist/load" \\
    -H "Content-Type: application/json" \\
    -d '{"tree_name": "default", "path": "trees/default.pkl"}'

echo ""
echo "Verifying tree loaded..."
curl -s "http://localhost:8000/api/v1/tree/stats" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Tree loaded: {d}')"

# Start the benchmark
echo ""
echo "Starting full benchmark (2556 queries)..."
cd ~/rag_benchmarking

nohup python3 scripts/run_multihop_eval.py \\
    --api-url http://localhost:8000 \\
    --max-queries 0 \\
    --concurrency 1 \\
    --top-k 10 \\
    --skip-ingest \\
    --output ~/benchmark_full_results.json > ~/benchmark.log 2>&1 &

BENCHMARK_PID=\$!
echo "Benchmark started with PID: \$BENCHMARK_PID"
echo "Estimated time: 8-10 hours"

# Save PIDs for monitoring
echo \$SERVER_PID > ~/server.pid
echo \$BENCHMARK_PID > ~/benchmark.pid
REMOTE_RUN

echo ""
echo "=========================================="
echo "          DEPLOYMENT COMPLETE            "
echo "=========================================="
echo ""
echo "Instance: $INSTANCE_ID"
echo "IP: $PUBLIC_IP"
echo ""
echo "Monitor progress:"
echo "  ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
echo "  tail -f ~/benchmark.log"
echo ""
echo "Check results when done:"
echo "  cat ~/benchmark_full_results.json"
echo ""
echo "Download results:"
echo "  scp -i $KEY_FILE ubuntu@$PUBLIC_IP:~/benchmark_full_results.json ."
echo ""
echo "Terminate instance when done:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
echo ""

# Save connection script
cat > /tmp/connect_benchmark.sh << EOF
#!/bin/bash
ssh -i $KEY_FILE ubuntu@$PUBLIC_IP
EOF
chmod +x /tmp/connect_benchmark.sh
echo "Quick connect: bash /tmp/connect_benchmark.sh"

# Save download script
cat > /tmp/download_results.sh << EOF
#!/bin/bash
scp -i $KEY_FILE ubuntu@$PUBLIC_IP:~/benchmark_full_results.json ./benchmark_full_results.json
scp -i $KEY_FILE ubuntu@$PUBLIC_IP:~/benchmark.log ./benchmark.log
echo "Results downloaded!"
EOF
chmod +x /tmp/download_results.sh
echo "Download results: bash /tmp/download_results.sh"
