import xml.etree.ElementTree as ET

from backend.policy.policy_model import Policy
from backend.xacml.xacml_generator import XACML_NS, generate_xacml


def test_xacml_is_parseable_and_escapes_values() -> None:
    xml = generate_xacml(Policy("Bác sĩ & nhóm", "xem", "hồ sơ <mật>"))
    root = ET.fromstring(xml)
    assert root.tag == f"{{{XACML_NS}}}Policy"
    assert root.find(f".//{{{XACML_NS}}}Rule").attrib["Effect"] == "Permit"
    values = root.findall(f".//{{{XACML_NS}}}AttributeValue")
    assert any(value.text == "Bác sĩ & nhóm" for value in values)
    assert any(value.text == "hồ sơ <mật>" for value in values)