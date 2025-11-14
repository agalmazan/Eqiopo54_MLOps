#!/bin/bash

# MLflow Server Setup Script for EC2
# This script installs and configures MLflow with Nginx reverse proxy

set -e  # Exit on any error

echo "=========================================="
echo "MLflow Server Setup Script"
echo "=========================================="

# Variables
MLFLOW_PORT=5000
MLFLOW_DATA_DIR="/data"
MLFLOW_DB_PATH="${MLFLOW_DATA_DIR}/mlflow.db"
S3_ARTIFACT_ROOT="s3://my-dvc-study-entry-performance-data/artifacts/"
NGINX_CONF="/etc/nginx/conf.d/mlflow.conf"
VENV_PATH="/home/ec2-user/mlflow_venv"

# Step 1: Setup swap space
echo "Step 1: Setting up swap space..."
if [ ! -f /swapfile ]; then
    sudo dd if=/dev/zero of=/swapfile bs=128M count=16
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# Step 2: Update system packages
echo "Step 2: Updating system packages..."
sudo yum update -y

# Step 3: Install Python 3 and pip
echo "Step 3: Installing Python 3 and pip..."
sudo yum install -y python3 python3-pip

# Step 4: Create and activate virtual environment
echo "Step 4: Creating Python virtual environment..."
rm -rf $VENV_PATH  # Clean up any existing venv
python3 -m venv $VENV_PATH
source $VENV_PATH/bin/activate

# Step 5: Install MLflow and dependencies
echo "Step 5: Installing MLflow and dependencies..."
# Upgrade pip first
pip3 install --upgrade pip

# Install packages
for package in wheel mlflow-skinny boto3 pymysql Flask pyarrow mlflow; do
    echo "Installing $package..."
    pip3 install --no-cache-dir $package
done

# Step 6: Install Nginx
echo "Step 6: Installing Nginx..."
sudo yum install -y nginx

# Step 7: Create data directory for MLflow
echo "Step 7: Creating MLflow data directory..."
sudo mkdir -p ${MLFLOW_DATA_DIR}
sudo chown ec2-user:ec2-user ${MLFLOW_DATA_DIR}

# Step 8: Verify S3 bucket access
echo "Step 8: Verifying S3 bucket access..."
if aws s3 ls ${S3_ARTIFACT_ROOT} > /dev/null 2>&1; then
    echo "✓ S3 bucket is accessible"
else
    echo "⚠ Warning: Cannot access S3 bucket. Please ensure:"
    echo "  1. The bucket exists: student-performance-mlflow"
    echo "  2. Your EC2 instance has an IAM role with S3 permissions"
    echo "  3. The IAM role has s3:PutObject, s3:GetObject, s3:ListBucket permissions"
fi

# Step 9: Create MLflow systemd service
echo "Step 9: Creating MLflow systemd service..."
sudo tee /etc/systemd/system/mlflow.service > /dev/null <<EOF
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user
Environment="PATH=/home/ec2-user/mlflow_venv/bin"
ExecStart=/home/ec2-user/mlflow_venv/bin/mlflow server \\
    --backend-store-uri sqlite:///${MLFLOW_DB_PATH} \\
    --default-artifact-root ${S3_ARTIFACT_ROOT} \\
    --host 0.0.0.0 \\
    --port ${MLFLOW_PORT} \\
    --serve-artifacts
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Step 10: Configure Nginx as reverse proxy
echo "Step 10: Configuring Nginx..."
sudo tee ${NGINX_CONF} > /dev/null <<'NGINXEOF'
server {
    listen 80;
    server_name _;

    # Increase timeouts for large artifact uploads
    client_max_body_size 500M;
    proxy_connect_timeout 600;
    proxy_send_timeout 600;
    proxy_read_timeout 600;
    send_timeout 600;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
NGINXEOF

# Step 11: Test Nginx configuration
echo "Step 11: Testing Nginx configuration..."
sudo nginx -t

# Step 12: Enable and start services
echo "Step 12: Starting services..."

# Start MLflow
sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow

# Start Nginx
sudo systemctl enable nginx
sudo systemctl restart nginx

# Step 13: Wait for MLflow to start
echo "Step 13: Waiting for MLflow to start..."
sleep 5

# Step 14: Verify services are running
echo "Step 14: Verifying services..."
echo ""
echo "MLflow Service Status:"
sudo systemctl status mlflow --no-pager | head -10
echo ""
echo "Nginx Service Status:"
sudo systemctl status nginx --no-pager | head -10

# Step 15: Get public IP
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
echo ""
echo "✓ MLflow is running on port ${MLFLOW_PORT}"
echo "✓ Nginx is proxying requests on port 80"
echo "✓ Artifacts will be stored in: ${S3_ARTIFACT_ROOT}"
echo ""
echo "Access MLflow UI at: http://${PUBLIC_IP}"
echo ""
echo "Useful commands:"
echo "  - Check MLflow logs: sudo journalctl -u mlflow -f"
echo "  - Check Nginx logs: sudo tail -f /var/log/nginx/error.log"
echo "  - Restart MLflow: sudo systemctl restart mlflow"
echo "  - Restart Nginx: sudo systemctl restart nginx"
echo ""
