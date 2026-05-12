#!/usr/bin/env bash
# ── 配置 ──────────────────────────────────────────────────────────────
CONDA_PYTHON="D:/Anaconda3/envs/py313/python.exe"
BACKEND_PORT=8000
FRONTEND_PORT=8501

# ── 清理旧进程 ────────────────────────────────────────────────────────
echo "清理旧进程..."
# 杀掉占用端口的旧进程
PYTHON_PID=$(netstat -ano 2>/dev/null | grep ":$BACKEND_PORT" | grep LISTEN | awk '{print $5}' | sort -u)
if [ -n "$PYTHON_PID" ]; then
  echo "  端口 $BACKEND_PORT 被 PID $PYTHON_PID 占用，正在清理..."
  kill -f "$PYTHON_PID" 2>/dev/null || true
  sleep 1
fi

# 杀掉残留的 run_server 进程
OLD_PIDS=$(tasklist //FI "IMAGENAME eq python.exe" //NH 2>/dev/null | grep -i run_server | awk '{print $2}' || true)
for pid in $OLD_PIDS; do
  echo "  杀掉残留进程 PID: $pid"
  kill -f "$pid" 2>/dev/null || true
done
sleep 1

# ── 启动后端（后台运行）───────────────────────────────────────────────
echo "启动后端 API 服务..."
PYTHONPATH="E:/agent-learning/multimodal-rag-agent-forAgri" \
  "$CONDA_PYTHON" run_server.py --reload &
BACKEND_PID=$!
echo "  后端 PID: $BACKEND_PID"

# 等待后端就绪
echo "等待后端就绪..."
for i in $(seq 1 15); do
  sleep 1
  if curl -s http://localhost:$BACKEND_PORT/health > /dev/null 2>&1; then
    echo "  后端已就绪 ✓"
    break
  fi
  if [ "$i" -eq 15 ]; then
    echo "  ⚠ 后端启动超时，继续尝试启动前端..."
  fi
done

# ── 启动前端 ──────────────────────────────────────────────────────────
echo "启动 Streamlit 前端 (端口 $FRONTEND_PORT) ..."
"$CONDA_PYTHON" -m streamlit run frontend/app.py
RC=$?

# ── 退出清理 ──────────────────────────────────────────────────────────
echo "正在关闭后端 (PID: $BACKEND_PID)..."
kill $BACKEND_PID 2>/dev/null || true
wait $BACKEND_PID 2>/dev/null || true
echo "服务已关闭。"
exit $RC