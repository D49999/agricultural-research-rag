"""
FastAPI 服务器启动入口。

用法
────
  # 开发模式（热重载）
  python run_server.py

  # 生产模式
  python run_server.py --host 0.0.0.0 --port 8000 --workers 4
"""
import argparse
import asyncio
import sys

import uvicorn

from config.settings import get_settings

settings = get_settings()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动农业科研 RAG API 服务器。")
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reload", action="store_true", help="启用热重载（开发模式）。")
    return parser.parse_args()


def _loop_setting() -> str | type[asyncio.AbstractEventLoop]:
    """
    Windows 上的事件循环选择。

    uvicorn 在 Windows 默认用 Proactor 事件循环。本项目查询接口是同步阻塞的
    （一次问答可能跑几十秒），阻塞结束后 IOCP 里堆积的失效连接会让 accept
    任务抛 OSError(WinError 64) 并退出监听循环——进程还在，但端口不再接受
    新连接（前端表现为「无法连接到远程服务器」）。本项目不使用子进程，
    改用 Selector 事件循环可避免该问题。
    """
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop
    return "auto"


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level=settings.log_level.lower(),
        loop=_loop_setting(),
    )
