"""Generate structurally valid XACML XML."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

from .policy_model import Policy

logger = logging.getLogger(__name__)

XACML_NS = "urn:oasis:names:tc:xacml:3.0:core:schema:wd-17"
ET.register_namespace("", XACML_NS)


def generate_xacml(policy: Policy) -> str:
    """Serialize an ABAC policy as XACML using ElementTree escaping."""
    qualified = lambda name: f"{{{XACML_NS}}}{name}"
    root = ET.Element(
        qualified("Policy"),
        {
            "PolicyId": "generated-abac-policy",
            "Version": "1.0",
            "RuleCombiningAlgId": "urn:oasis:names:tc:xacml:1.0:rule-combining-algorithm:first-applicable",
        },
    )
    target = ET.SubElement(root, qualified("Target"))
    any_of = ET.SubElement(target, qualified("AnyOf"))
    all_of = ET.SubElement(any_of, qualified("AllOf"))

    for category, attribute_id, value in (
        ("access-subject", "subject-id", policy.subject),
        ("action", "action-id", policy.action),
        ("resource", "resource-id", policy.object),
    ):
        match = ET.SubElement(
            all_of,
            qualified("Match"),
            {"MatchId": "urn:oasis:names:tc:xacml:1.0:function:string-equal"},
        )
        ET.SubElement(match, qualified("AttributeValue"), {"DataType": "http://www.w3.org/2001/XMLSchema#string"}).text = value
        ET.SubElement(
            match,
            qualified("AttributeDesignator"),
            {
                "AttributeId": attribute_id,
                "Category": f"urn:oasis:names:tc:xacml:3.0:attribute-category:{category}",
                "DataType": "http://www.w3.org/2001/XMLSchema#string",
                "MustBePresent": "false",
            },
        )

    ET.SubElement(root, qualified("Rule"), {"RuleId": "generated-rule", "Effect": policy.effect}).append(ET.Element(qualified("Description")))
    rule = root.find(qualified("Rule"))
    if rule is not None:
        description = rule.find(qualified("Description"))
        if description is not None:
            description.text = "Generated from a Vietnamese ACP sentence"
    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode", xml_declaration=True)
    logger.info("XACML generated")
    return xml