"""XACML 3.0 XML Policy generator."""

from xml.etree import ElementTree as ET
from typing import Any

XACML_NS = "urn:oasis:names:tc:xacml:3.0:core:schema:wd-17"


def generate_xacml(policy: Any) -> str:
    """Generate XACML XML Policy representation from Policy object."""
    policy_id = f"Policy_{policy.subject.replace(' ', '_')}_{policy.action}"
    
    root = ET.Element("Policy", {
        "xmlns": "urn:oasis:names:tc:xacml:3.0:core:schema:wd-17",
        "PolicyId": policy_id,
        "RuleCombiningAlgId": "urn:oasis:names:tc:xacml:1.0:rule-combining-algorithm:first-applicable",
        "Version": "1.0"
    })
    
    rule = ET.SubElement(root, "Rule", {
        "RuleId": f"Rule_{policy.action}",
        "Effect": policy.effect
    })
    
    target = ET.SubElement(rule, "Target")
    
    # Subject Match
    subj_match = ET.SubElement(target, "AnyOf")
    all_subj = ET.SubElement(subj_match, "AllOf")
    m_subj = ET.SubElement(all_subj, "Match", {"MatchId": "urn:oasis:names:tc:xacml:1.0:function:string-equal"})
    ET.SubElement(m_subj, "AttributeValue", {"DataType": "http://www.w3.org/2001/XMLSchema#string"}).text = policy.subject
    ET.SubElement(m_subj, "AttributeDesignator", {
        "Category": "urn:oasis:names:tc:xacml:1.0:subject-category:access-subject",
        "AttributeId": "urn:oasis:names:tc:xacml:1.0:subject:subject-id",
        "DataType": "http://www.w3.org/2001/XMLSchema#string"
    })
    
    # Action Match
    act_match = ET.SubElement(target, "AnyOf")
    all_act = ET.SubElement(act_match, "AllOf")
    m_act = ET.SubElement(all_act, "Match", {"MatchId": "urn:oasis:names:tc:xacml:1.0:function:string-equal"})
    ET.SubElement(m_act, "AttributeValue", {"DataType": "http://www.w3.org/2001/XMLSchema#string"}).text = policy.action
    ET.SubElement(m_act, "AttributeDesignator", {
        "Category": "urn:oasis:names:tc:xacml:3.0:attribute-category:action",
        "AttributeId": "urn:oasis:names:tc:xacml:1.0:action:action-id",
        "DataType": "http://www.w3.org/2001/XMLSchema#string"
    })
    
    # Resource/Object Match
    obj_match = ET.SubElement(target, "AnyOf")
    all_obj = ET.SubElement(obj_match, "AllOf")
    m_obj = ET.SubElement(all_obj, "Match", {"MatchId": "urn:oasis:names:tc:xacml:1.0:function:string-equal"})
    ET.SubElement(m_obj, "AttributeValue", {"DataType": "http://www.w3.org/2001/XMLSchema#string"}).text = policy.object
    ET.SubElement(m_obj, "AttributeDesignator", {
        "Category": "urn:oasis:names:tc:xacml:3.0:attribute-category:resource",
        "AttributeId": "urn:oasis:names:tc:xacml:1.0:resource:resource-id",
        "DataType": "http://www.w3.org/2001/XMLSchema#string"
    })
    
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8").decode("utf-8")
