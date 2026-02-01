#!/bin/bash
# Setup and run RAG benchmark on EC2
# This script runs ON the EC2 instance

set -e

echo "=== RAG Benchmark Setup ==="
echo "Started at: $(date)"

# Update system
echo "Updating system..."
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git

# Clone repositories
echo "Cloning repositories..."
cd ~

if [ ! -d "incidentfox" ]; then
    git clone https://github.com/incidentfox/incidentfox.git
fi

if [ ! -d "rag_benchmarking" ]; then
    git clone https://github.com/YOUR_USERNAME/rag_benchmarking.git || \
    echo "rag_benchmarking repo not found, will copy from local"
fi

# Setup incidentfox
echo "Setting up incidentfox..."
cd ~/incidentfox
git checkout fix/teach-response-and-retrieval-chunk
git pull

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install sentence-transformers umap-learn

# Move trees if uploaded to home
if [ -d ~/trees ]; then
    echo "Moving tree data..."
    mv ~/trees ~/incidentfox/
fi

# Setup benchmarking
echo "Setting up rag_benchmarking..."
cd ~/rag_benchmarking 2>/dev/null || cd ~
pip install requests tqdm

# Set API key (will be set by environment or prompt)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set!"
    echo "Set it with: export OPENAI_API_KEY='your-key'"
    exit 1
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run the benchmark:"
echo "  cd ~/incidentfox && source .venv/bin/activate"
echo "  nohup bash ~/run_benchmark.sh > benchmark.log 2>&1 &"
echo ""
echo "Or run the full benchmark script now? (y/n)"
