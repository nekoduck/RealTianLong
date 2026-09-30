"""
[INPUT]: 已安装的游戏网页依赖与官方 cloudflared CLI；标准库 subprocess / urllib
[OUTPUT]: 启动本地游戏与免费的 HTTPS 外网试玩隧道，写 .cache/public/deployment.json 供重新查找入口
[POS]: 开发部署工具。游戏只监听本机；隧道只代理这个端口；Ctrl+C/TERM 同时关闭两个子进程。
       使用 JSON 回合接口，因为 Quick Tunnel 不支持 SSE。不是常驻云服务器，电脑需保持在线。
       --allow-migration 显式将迁移选择传给游戏服务；已有存档需事先备份。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def main() -> int:
    ap = argparse.ArgumentParser(description="部署 RealTianLong 外网试玩环境（本机需在线）")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--allow-migration", action="store_true", help="显式用当前规则接续此部署的旧存档")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cloudflared = shutil.which("cloudflared")
    if not cloudflared:
        ap.error("请先从 Cloudflare 官方渠道安装 cloudflared")
    if not 1 <= args.port <= 65535:
        ap.error("端口必须在 1..65535 之间")
    # 不能把已有的其他本机服务误代理到外网。
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
    except OSError:
        ap.error("端口已被占用，请使用 --port 另选端口")
    directory = root / ".cache" / "public"
    directory.mkdir(parents=True, exist_ok=True)
    children: list[subprocess.Popen] = []

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        with (directory / "game.log").open("a") as output:
            command = [sys.executable, "-m", "tianlong.runtime.web", "--port", str(args.port),
                       "--transport", "json", "--data-dir", str(root / ".cache" / "web-saves")]
            if args.allow_migration:
                command.append("--allow-migration")
            game = subprocess.Popen(command,
                                    cwd=root, stdout=output, stderr=output)
        children.append(game)
        for _ in range(120):
            if game.poll() is not None:
                raise RuntimeError("游戏服务未启动，请检查 .cache/public/game.log")
            try:
                with urlopen(f"http://127.0.0.1:{args.port}/healthz", timeout=1) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                time.sleep(0.25)
        else:
            raise RuntimeError("游戏服务启动超时")
        tunnel = subprocess.Popen([cloudflared, "tunnel", "--config", "/dev/null", "--no-autoupdate",
                                   "--protocol", "http2", "--url", f"http://127.0.0.1:{args.port}"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        children.append(tunnel)
        with (directory / "tunnel.log").open("w") as output:
            for line in tunnel.stdout:
                output.write(line)
                output.flush()
                match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                if match:
                    record = {"url": match[0], "pid": os.getpid(), "game_pid": game.pid,
                              "tunnel_pid": tunnel.pid, "port": args.port, "transport": "json"}
                    (directory / "deployment.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
                    print(f"外网试玩入口：{match[0]}（本机需保持在线）", flush=True)
                if game.poll() is not None:
                    raise RuntimeError("游戏进程已停止")
        raise RuntimeError("外网隧道已停止，请检查 .cache/public/tunnel.log")
    except KeyboardInterrupt:
        return 0
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
