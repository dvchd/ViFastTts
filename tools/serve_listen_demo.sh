#!/bin/bash
# tools/serve_listen_demo.sh — phục vụ một thư mục demo nghe qua Cloudflare quick tunnel.
#
# Dùng: tools/serve_listen_demo.sh [port] [dir] [log]
#   mặc định: 8788 · artifacts/listen_small_filt · artifacts/listen_small_filt/serve.log
# Chạy nền bền (không chết theo phiên terminal):
#   launchctl submit -l vifasttts.listen -- /bin/bash "$PWD/tools/serve_listen_demo.sh" \
#       8788 artifacts/listen_small_filt artifacts/listen_small_filt/serve.log
# Gỡ khi xong: launchctl remove vifasttts.listen
#
# Lý do tồn tại: người dùng remote (điện thoại) cần nghe demo qua internet — quick tunnel cho URL
# https://*.trycloudflare.com tạm thời, không cần đăng nhập Cloudflare, không đổi DNS.
set -u
cd "$(dirname "$0")/.." || exit 1
PORT="${1:-8788}"
DIR="${2:-artifacts/listen_small_filt}"
LOG="${3:-$DIR/serve.log}"
mkdir -p "$DIR"
touch "$LOG"

# Dọn server cũ cùng cổng (trường hợp lần chạy trước bị kill cứng, còn orphan).
pkill -f "http.server $PORT" 2>/dev/null || true
sleep 1

# Server tĩnh nội bộ — CHỈ localhost; tunnel là cửa ra internet duy nhất.
.venv/bin/python -m http.server "$PORT" --bind 127.0.0.1 --directory "$DIR" >> "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT
echo "[serve] $(date '+%F %H:%M:%S') http://127.0.0.1:$PORT → $DIR (pid $SERVER_PID)" >> "$LOG"

# Tunnel công khai (URL in vào log; đổi mỗi lần chạy).
/opt/homebrew/bin/cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate >> "$LOG" 2>&1
echo "[serve] $(date '+%F %H:%M:%S') tunnel kết thúc" >> "$LOG"
