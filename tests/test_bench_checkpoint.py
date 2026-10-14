"""Test cổng của `tools/bench_checkpoint.py` (§44 Lượt 1b): tương đối, ba trạng thái."""

from __future__ import annotations

from tools.bench_checkpoint import FAIL, NO_BASE, NO_DATA, PASS, hard_gates, target_gates

BASE_OLD = {"mel_l1_z": 1.14, "mel_db_speech": 8.9, "rough": 0.42, "tone_C": 0.80,
            "cer": 0.23, "f0_nga": 177.0}
BASE_FIX = {"mel_gap": 3.1, "pause_f1_all": 0.50, "gap_sum_mae_all": 200.0,
            "syl_z": 1.16, "mel_gap_z": 3.0}


def _row_old(**over):
    row = {"mel_l1_z": 1.14, "mel_db_speech": 8.95, "tone_C": 0.80, "cer": 0.24, "f0_nga": 178.0}
    row.update(over)
    return row


def _row_fix(**over):
    row = {"mel_gap": 3.0, "pause_f1_all": 0.58, "gap_sum_mae_all": 140.0, "length_ratio": 1.02,
           "syl_z": 1.16, "mel_gap_z": 2.5}
    row.update(over)
    return row


def test_cong_cung_co_syl_z_va_muc_tieu_gap_z():
    """§44/B3: syl_z là cổng cứng (≤ mốc×1.01); gap_z là mục tiêu (≤ mốc×0.85)."""
    base_fix = {**BASE_FIX, "syl_z": 1.16, "mel_gap_z": 3.0}
    row_fix = {**_row_fix(), "syl_z": 1.17, "mel_gap_z": 2.5}
    hard = hard_gates(_row_old(), BASE_OLD, row_fix, base_fix, rough_old=0.42)
    assert hard["syl_z"][0] == PASS
    bad = hard_gates(_row_old(), BASE_OLD, {**row_fix, "syl_z": 1.19}, base_fix, rough_old=0.42)
    assert bad["syl_z"][0] == FAIL
    assert target_gates(row_fix, base_fix)["gap_z≤mốc×0.85"][0] == PASS
    assert target_gates({**row_fix, "mel_gap_z": 2.6}, base_fix)["gap_z≤mốc×0.85"][0] == FAIL


def test_cong_cung_tuong_doi_dung_chieu():
    """mel_z ≤ mốc×1.01 · dB ≤ mốc+0.10 · rough ≥ mốc−0.01 · tone ≥ mốc−0.01 · CER ≤ mốc+0.03 · F0 ≤ mốc+5."""
    gates = hard_gates(_row_old(), BASE_OLD, _row_fix(), BASE_FIX, rough_old=0.42)
    assert all(state == PASS for state, _ in gates.values()), gates
    bad = hard_gates(_row_old(mel_l1_z=1.16, cer=0.27, f0_nga=183.0), BASE_OLD, _row_fix(mel_gap=3.2),
                     BASE_FIX, rough_old=0.40)
    assert bad["mel_l1_z"][0] == FAIL and bad["cer"][0] == FAIL
    assert bad["f0_nga"][0] == FAIL and bad["mel_gap"][0] == FAIL


def test_cong_cung_thieu_du_lieu_la_truot():
    """A2: cổng cứng thiếu dữ liệu (row None) = TRƯỢT kèm 'thiếu dữ liệu'."""
    gates = hard_gates(_row_old(mel_l1_z=None), BASE_OLD, _row_fix(mel_gap=None), BASE_FIX, rough_old=None)
    assert gates["mel_l1_z"] == (FAIL, "thiếu dữ liệu")
    assert gates["rough"][0] == FAIL and gates["mel_gap"][0] == FAIL


def test_cong_cung_thieu_moc_la_chua_co_moc():
    """A3: thiếu mốc = 'chưa có mốc' (không phải trượt, không phải đạt)."""
    gates = hard_gates(_row_old(), None, _row_fix(), None, rough_old=0.42)
    assert all(state == NO_BASE for state, _ in gates.values())
    assert gates["mel_l1_z"][1] == "chưa có mốc"


def test_cong_muc_tieu_thieu_du_lieu_la_chua_do():
    """A3: cổng mục tiêu thiếu dữ liệu = 'chưa đo' (không tính là trượt)."""
    gates = target_gates(_row_fix(pause_f1_all=None), BASE_FIX)
    assert gates["pause_f1_all≥mốc+0.073"] == (NO_DATA, "chưa đo")
    assert gates["gap_sum_mae_all≤mốc×0.714"][0] == PASS
    assert gates["length_ratio 1.00±0.05"][0] == PASS
    bad = target_gates(_row_fix(pause_f1_all=0.55, gap_sum_mae_all=150.0, length_ratio=1.2,
                                mel_gap_z=2.6), BASE_FIX)
    assert all(state == FAIL for state, _ in bad.values())


def test_ket_luan_bon_truong_hop():
    """A4: thứ tự ưu tiên TỪ CHỐI → TRƯỢT MỤC TIÊU → CHƯA ĐỦ SỐ → CHẤP NHẬN."""
    from tools.bench_checkpoint import conclusion

    hard_pass = {"mel_l1_z": (PASS, "")}
    hard_fail = {"mel_l1_z": (FAIL, "")}
    hard_nobase = {"mel_l1_z": (NO_BASE, "chưa có mốc")}
    target_pass = {"pause_f1_all≥mốc+0.073": (PASS, "")}
    target_fail = {"pause_f1_all≥mốc+0.073": (FAIL, "")}
    target_pending = {"pause_f1_all≥mốc+0.073": (NO_DATA, "chưa đo")}
    assert conclusion(hard_fail, target_pass) == "TỪ CHỐI"
    assert conclusion(hard_nobase, target_pass) == "TỪ CHỐI"  # thiếu mốc = coi như trượt
    assert conclusion(hard_pass, target_fail) == "ĐẠT CỨNG, TRƯỢT MỤC TIÊU"
    assert conclusion(hard_pass, target_pending) == "ĐẠT CỨNG, MỤC TIÊU CHƯA ĐỦ SỐ"
    assert conclusion(hard_pass, target_pass) == "CHẤP NHẬN"


def test_run_tool_xoa_tep_cu_va_khong_doc_khi_loi(tmp_path):
    """C8: tệp cũ cùng tên + subprocess lỗi ⇒ trả None (và tệp cũ đã bị xoá)."""

    from tools.bench_checkpoint import _run_tool

    out = tmp_path / "ket_qua.json"
    out.write_text('{"cu": true}', encoding="utf-8")
    result = _run_tool(["-c", "raise SystemExit(1)"], out, "giả lỗi")
    assert result is None
    assert not out.exists()


def test_summary_fixgap_khong_co_tone_creak_cer():
    """C9/V10.A6: hàng val fixtail KHÔNG chứa tone_C/creak/cer (kể cả dạng null)."""
    import json
    from pathlib import Path

    data = json.loads(Path("eval/baselines/base40k_fixtail.json").read_text(encoding="utf-8"))
    for key in ("tone_C", "creak_djitter_nga", "creak_dhnr_nga", "cer"):
        assert key not in data["summary_fixgap"], key


def test_moc_chuan_hoa_lf_sort_keys():
    """V10.A3: tệp mốc LF + sort_keys + indent 2 + dòng trống cuối (sha ổn định giữa các máy)."""
    import json
    from pathlib import Path

    for name in ("base40k_old.json", "base40k_fixtail.json"):
        raw = Path("eval/baselines", name).read_bytes()
        assert b"\r" not in raw, name
        assert raw.endswith(b"\n"), name
        data = json.loads(raw.decode("utf-8"))
        expected = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        assert raw.decode("utf-8") == expected, name
        for section in ("summary", "summary_fixgap"):
            for key, value in data.get(section, {}).items():
                assert value is not None, f"{name} {section} {key} = null"


def test_rough_from_history_bo_dong_chu_thich(tmp_path):
    """C10: history.csv có dòng '# X3 resume: …' trước header vẫn đọc ra rough đúng."""
    from tools.bench_checkpoint import _rough_from_history

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "history.csv").write_text(
        "# X3 resume: resumed_from=a@40000, val_hash_old=x, val_hash_new=y\n"
        "step,phase,train_total,train_mel,mel_gap,mel_voiced,val_total,val_mel_voiced,val_rough,lr\n"
        "45000,B,5.0,0.5,0.6,0.55,8.0,0.58,0.41,1e-5\n"
        "50000,B,5.1,0.5,0.6,0.55,8.1,0.59,0.43,1e-5\n", encoding="utf-8")
    assert _rough_from_history(run_dir, 50000) == 0.43
    assert _rough_from_history(run_dir, 45000) == 0.41


def test_check_baseline_pin_lech_thi_dung(tmp_path):
    """B5: sha256 tệp mốc khác giá trị ghim trong README ⇒ SystemExit; khớp ⇒ im lặng."""
    import hashlib

    import pytest as _pytest

    from tools.bench_checkpoint import check_baseline_pin

    target = tmp_path / "moc.json"
    target.write_text("{}", encoding="utf-8")
    good = hashlib.sha256(target.read_bytes()).hexdigest()
    readme = tmp_path / "README.md"
    readme.write_text(f"| `moc.json` | `{good}` | dùng cho cổng |\n", encoding="utf-8")
    check_baseline_pin(str(target), readme)  # khớp ⇒ không lỗi
    target.write_text('{"khac": 1}', encoding="utf-8")
    with _pytest.raises(SystemExit, match="sha256"):
        check_baseline_pin(str(target), readme)


def test_pin_readme_khop_byte_trong_git():
    """V10.A3: sha tệp mốc theo byte TRONG GIT == sha ghim README (mọi lần test) · byte đĩa == byte git."""
    import hashlib
    import subprocess
    from pathlib import Path

    from tools.bench_checkpoint import readme_sha256

    root = Path(__file__).resolve().parents[1]
    for name in ("base40k_old.json", "base40k_fixtail.json"):
        rel = f"eval/baselines/{name}"
        blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=root, capture_output=True)
        assert blob.returncode == 0, f"{rel} chưa có trong HEAD — commit trước khi ghim"
        git_sha = hashlib.sha256(blob.stdout).hexdigest()
        pinned = readme_sha256(name, root / "eval/README.md")
        assert pinned == git_sha, f"{name}: README ghim {pinned} ≠ byte trong git {git_sha} — ghim lại"
        disk = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        assert disk == git_sha, f"{name}: byte trên đĩa ≠ byte trong git — checkout cũ, chạy: git checkout -- {rel}"
