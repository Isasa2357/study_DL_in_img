from __future__ import annotations

import sys
import traceback


def section(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def ng(msg: str) -> None:
    print(f"[NG] {msg}")


def main() -> int:
    section("Import")
    try:
        import torch
        import torchvision
        import torchaudio

        ok(f"Python        : {sys.version}")
        ok(f"torch         : {torch.__version__}")
        ok(f"torchvision   : {torchvision.__version__}")
        ok(f"torchaudio    : {torchaudio.__version__}")
    except Exception:
        ng("import に失敗しました")
        traceback.print_exc()
        return 1

    section("CUDA")
    try:
        print(f"torch.version.cuda           : {torch.version.cuda}")
        print(f"torch.cuda.is_available()    : {torch.cuda.is_available()}")
        print(f"torch.backends.cudnn.enabled : {torch.backends.cudnn.enabled}")

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            ok(f"CUDA device count: {device_count}")
            for i in range(device_count):
                print(f"  GPU[{i}] : {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA は使えません。CPU のみ確認します。")
    except Exception:
        ng("CUDA 情報の取得に失敗しました")
        traceback.print_exc()
        return 1

    section("Torch basic test")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.randn(1024, 1024, device=device)
        y = torch.randn(1024, 1024, device=device)
        z = x @ y

        if device == "cuda":
            torch.cuda.synchronize()

        ok(f"torch のテンソル演算成功: device={device}, shape={tuple(z.shape)}")
    except Exception:
        ng("torch の基本演算に失敗しました")
        traceback.print_exc()
        return 1

    section("torchvision test")
    try:
        import torch
        import torchvision

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torchvision.models.resnet18(weights=None).to(device)
        model.eval()

        dummy = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            out = model(dummy)

        if device == "cuda":
            torch.cuda.synchronize()

        ok(f"torchvision モデル実行成功: output shape={tuple(out.shape)}")
    except Exception:
        ng("torchvision のテストに失敗しました")
        traceback.print_exc()
        return 1

    section("torchaudio test")
    try:
        import torch
        import torchaudio

        sample_rate = 16000
        duration_sec = 1.0
        num_samples = int(sample_rate * duration_sec)

        waveform = torch.randn(1, num_samples)  # [channels, time]
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)
        mel = transform(waveform)

        ok(f"torchaudio 変換成功: waveform={tuple(waveform.shape)}, mel={tuple(mel.shape)}")
    except Exception:
        ng("torchaudio のテストに失敗しました")
        traceback.print_exc()
        return 1

    section("Result")
    ok("torch / torchvision / torchaudio は正常に使えています")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())