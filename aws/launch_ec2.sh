#!/bin/bash
# Launch EC2 instance for RAG benchmarking
# Usage: ./launch_ec2.sh

set -e

# Configuration - UPDATE THESE
KEY_NAME="${AWS_KEY_NAME:-your-key-pair}"  # Your SSH key pair name
SECURITY_GROUP="${AWS_SECURITY_GROUP:-}"   # Security group ID (leave empty to create new)
REGION="${AWS_REGION:-us-west-2}"

# Instance config
INSTANCE_TYPE="m5.xlarge"  # 4 vCPU, 16GB RAM - good balance
AMI_ID="ami-055c254ebd87b4dba"  # Ubuntu 22.04 LTS in us-west-2

echo "=== RAG Benchmark EC2 Launcher ==="
echo "Region: $REGION"
echo "Instance type: $INSTANCE_TYPE"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "ERROR: AWS CLI not installed. Install with: brew install awscli"
    exit 1
fi

# Check credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "ERROR: AWS credentials not configured. Run: aws configure"
    exit 1
fi

# Create security group if needed
if [ -z "$SECURITY_GROUP" ]; then
    echo "Creating security group..."
    SECURITY_GROUP=$(aws ec2 create-security-group \
        --group-name "rag-benchmark-sg" \
        --description "Security group for RAG benchmark" \
        --query 'GroupId' \
        --output text 2>/dev/null || \
        aws ec2 describe-security-groups \
            --group-names "rag-benchmark-sg" \
            --query 'SecurityGroups[0].GroupId' \
            --output text)
    
    # Allow SSH
    aws ec2 authorize-security-group-ingress \
        --group-id "$SECURITY_GROUP" \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 2>/dev/null || true
    
    echo "Security group: $SECURITY_GROUP"
fi

# Launch instance
echo "Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=rag-benchmark}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance ID: $INSTANCE_ID"
echo "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "=== Instance Ready ==="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo ""
echo "Wait 2-3 minutes for instance to fully boot, then:"
echo ""
echo "1. SSH into instance:"
echo "   ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "2. Upload tree data:"
echo "   scp -i ~/.ssh/${KEY_NAME}.pem -r /Users/apple/Desktop/incidentfox/trees ubuntu@${PUBLIC_IP}:~/"
echo ""
echo "3. Run setup and benchmark:"
echo "   ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'bash -s' < aws/setup_and_run.sh"
echo ""

# Save instance info
echo "$INSTANCE_ID" > /tmp/rag_benchmark_instance.txt
echo "$PUBLIC_IP" > /tmp/rag_benchmark_ip.txt

echo "Instance info saved to /tmp/rag_benchmark_*.txt"
echo ""
echo "To terminate when done:"
echo "   aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
