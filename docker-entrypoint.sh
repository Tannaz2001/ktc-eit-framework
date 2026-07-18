#!/bin/bash
set -e

# Get the local IP
LOCAL_IP=$(hostname -I | awk '{print $1}')

# Display startup message
echo ""
echo "=================================================="
echo "🚀 KTC Dashboard Starting..."
echo "=================================================="
echo ""
echo "📍 Local Access:"
echo "   http://localhost:8501"
echo ""
echo "📍 Network Access (from other devices):"
echo "   http://${LOCAL_IP}:8501"
echo ""
echo "=================================================="
echo ""

# Start Streamlit
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
