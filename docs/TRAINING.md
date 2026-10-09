# Quy trình huấn luyện

## Giai đoạn 1

Train iSTFTNet bằng mel thật. Train thêm HiFi-GAN baseline.

## Giai đoạn 2

Train acoustic model với duration, F0 và energy thật.

## Giai đoạn 3

Train predictors.

## Giai đoạn 4

Suy luận hoàn toàn bằng predicted features.

## Giai đoạn 5

Joint fine-tuning acoustic model và vocoder.

## Giai đoạn 6

Export ONNX, kiểm tra tương đương, sau đó mới FP16 và INT8.

## Cảnh báo

Repository không thể tự tạo alignment target hoàn hảo cho mọi dataset. Bạn cần tích hợp aligner phù hợp với dữ liệu, sau đó cache duration, mel, F0 và energy trước khi chạy optimizer loop production.
