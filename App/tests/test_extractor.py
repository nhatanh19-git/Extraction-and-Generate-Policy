from backend.extractor import AttributeExtractor
from backend.nlp_engine import NLPEngine


def test_required_vietnamese_sentences() -> None:
    cases = {
        "Bác sĩ có thể xem hồ sơ bệnh án.": ("Bác sĩ", "xem", "hồ sơ bệnh án"),
        "Quản trị viên được phép xóa tài khoản người dùng.": ("Quản trị viên", "xóa", "tài khoản người dùng"),
        "Nhân viên có thể tải báo cáo tài chính.": ("Nhân viên", "tải", "báo cáo tài chính"),
        "Giảng viên được phép xem điểm của sinh viên.": ("Giảng viên", "xem", "điểm của sinh viên"),
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