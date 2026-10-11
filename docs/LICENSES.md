# Ghi chú giấy phép các thành phần tham chiếu

ViFastTts (MIT) **không** nhúng code của các hệ G2P bên ngoài. Tài liệu này ghi lại giấy phép của các thành phần được tham chiếu/nghiên cứu trong `docs/G2P_VIETNAMESE.md` và `docs/TTS_ARCHITECTURE_RESEARCH.md`, để quyết định tích hợp về sau.

## Có thể dùng cho công cụ/tham chiếu (kể cả thương mại)

| Thành phần | Giấy phép | Ghi chú |
| --- | --- | --- |
| sea-g2p | Apache-2.0 | Rust core + C ABI; dùng cho audit/lexicon build-time |
| Epitran | MIT (PyPI: MIT-Modern-Variant) | Map `vie-Latn`; cần bật `tones=True` |
| Montreal Forced Aligner (tool) | MIT | Alignment/G2P build-time |
| MFA models G2P/acoustic tiếng Việt | CC BY 4.0 | Ghi nguồn khi dùng |
| WikiPron (code) | Apache-2.0 | **Dữ liệu** theo điều khoản Wiktionary (thường CC BY-SA) |
| ViMFA, CharsiuG2P | MIT | Charsiu dùng ByT5 (Apache-2.0) |

## Chỉ dùng offline/đối chiếu, KHÔNG nhúng runtime

| Thành phần | Giấy phép | Ghi chú |
| --- | --- | --- |
| vPhon | GPL-3.0 | Rule-based 3 phương ngữ; hữu ích để audit OOV |
| phonemizer + eSpeak NG | GPL-3.0 | 3 giọng Việt; output không tách âm tiết |
| hoang1007/vig2p | Chưa công bố license | Phụ thuộc `cmudict`/`phonemizer` (GPL) |
| Viphoneme (viG2P gốc) | MIT template + ghi chú AILAB | Xác nhận riêng nếu dùng thương mại |

## Nguyên tắc của ViFastTts

- Runtime chỉ dùng code trong `src/vifasttts` (MIT) + dependency đã liệt kê trong `pyproject.toml`.
- Mọi công cụ GPL chỉ chạy trong môi trường nghiên cứu/audit ngoài repo, không import vào package.
- Dữ liệu giọng nói, lexicon và checkpoint do người dùng tạo có giấy phép riêng.

## Dependency runtime chính

| Thành phần | Giấy phép | Dùng cho |
| --- | --- | --- |
| numpy | BSD-3-Clause | mảng số |
| PyYAML | MIT | đọc config |
| Typer | MIT | CLI |
| PyTorch | BSD-3-Clause | model, train, export |
| onnx / onnxruntime | Apache-2.0 / MIT | export, inference, quantization |
| soundfile (tùy chọn) | BSD-3-Clause | đọc WAV có resample |
| pytest / ruff / mypy (dev) | MIT / MIT / MIT | kiểm thử, lint, type |

## Ngữ liệu chỉ dùng để phân tích (không vendor)

| Thành phần | Giấy phép | Ghi chú |
| --- | --- | --- |
| duyet/vietnamese-wordlist (Viet11K–74K) | GPL-2.0 | Chỉ chạy audit cục bộ (`tools/parser_audit.py --wordlist`), không sao chép vào repo/package |
| WikiPron vi (hanoi/hue/saigon) | Code Apache-2.0; dữ liệu theo Wiktionary (thường CC BY-SA) | Dùng build-time để kiểm G2P/đối chiếu; ghi nguồn khi dẫn kết quả |
