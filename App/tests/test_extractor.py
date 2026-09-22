from backend.extractor.attribute_extractor import AttributeExtractor
from backend.nlp.nlp_engine import NLPEngine
from backend.batch.batch_processor import BatchProcessor
from pathlib import Path
import json


def test_required_vietnamese_sentences() -> None:
    cases = {
        "Bác sĩ có thể xem hồ sơ bệnh án.": ("Bác sĩ", "xem", "hồ sơ bệnh án"),
        "Quản trị viên được phép xóa tài khoản người dùng.": ("Quản trị viên", "xóa", "tài khoản người dùng"),
        "Nhân viên có thể tải báo cáo tài chính.": ("Nhân viên", "tải", "báo cáo tài chính"),
        "Giảng viên được phép xem điểm của sinh viên.": ("Giảng viên", "xem", "điểm của sinh viên"),
        "Bác sĩ có thể cập nhật hồ sơ bệnh viện": ("Bác sĩ", "cập nhật", "hồ sơ bệnh viện"),
        "Sinh viên dân sự có thể xem điểm trên hệ thống": ("Sinh viên dân sự", "xem", "điểm"),
    }
    engine = NLPEngine()
    extractor = AttributeExtractor()

    for sentence, expected in cases.items():
        result = extractor.extract(engine.process(sentence))
        assert result.subject is not None
        assert result.action is not None
        assert result.object is not None
        assert (result.subject.text, result.action.text, result.object.text) == expected


def test_invalid_sentence_returns_partial_result_without_crashing() -> None:
    result = AttributeExtractor().extract(NLPEngine().process("abc xyz"))
    assert result.subject is None
    assert result.action is None
    assert result.object is None


def test_json_export_and_batch_processing(tmp_path: Path) -> None:
    txt_file = tmp_path / "policies.txt"
    txt_file.write_text("Bác sĩ có thể cập nhật hồ sơ bệnh viện\nSinh viên dân sự có thể xem điểm trên hệ thống\n", encoding="utf-8")
    
    out_json = tmp_path / "extracted.json"
    batch_proc = BatchProcessor()
    results = batch_proc.process_txt_file(txt_file, out_json)

    assert len(results) == 2
    assert results[0]["subject"] == "Bác sĩ"
    assert results[0]["action"] == "cập nhật"
    assert results[0]["object"] == "hồ sơ bệnh viện"

    assert results[1]["subject"] == "Sinh viên dân sự"
    assert results[1]["action"] == "xem"
    assert results[1]["object"] == "điểm"
    assert results[1]["environment"] == "trên hệ thống"