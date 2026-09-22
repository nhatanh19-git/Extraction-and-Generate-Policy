from backend.extractor.attribute_extractor import AttributeExtractor
from backend.nlp.nlp_engine import NLPEngine
from backend.policy.policy_generator import generate_policy


def test_policy_defaults_to_permit() -> None:
    extraction = AttributeExtractor().extract(NLPEngine().process("Bác sĩ có thể xem hồ sơ bệnh án."))
    policy = generate_policy(extraction)
    assert policy.subject == "Bác sĩ"
    assert policy.action == "xem"
    assert policy.object == "hồ sơ bệnh án"
    assert policy.effect == "Permit"