import xml.etree.ElementTree as ET

from app.api.cap_format import CAP_NAMESPACE, build_cap_xml


SAMPLE_ALERTS = [
    {
        "cell": (12, 34),
        "lat": 20.85,
        "lon": 78.9,
        "probability": 0.82,
        "lead_time_min": 30,
        "why": "Driven primarily by rising lightning strike density (+0.40 over the window).",
    },
    {
        "cell": (5, 6),
        "lat": 21.1,
        "lon": 79.0,
        "probability": 0.45,
        "lead_time_min": 90,
        "why": "",
    },
]


def test_build_cap_xml_is_well_formed():
    xml_str = build_cap_xml(SAMPLE_ALERTS)
    root = ET.fromstring(xml_str)  # raises if not well-formed XML
    assert root.tag.endswith("alert")


def test_build_cap_xml_status_is_always_draft():
    xml_str = build_cap_xml(SAMPLE_ALERTS)
    root = ET.fromstring(xml_str)
    ns = {"cap": CAP_NAMESPACE}
    status = root.find("cap:status", ns)
    assert status is not None
    assert status.text == "Draft"


def test_build_cap_xml_one_info_block_per_alert():
    xml_str = build_cap_xml(SAMPLE_ALERTS)
    root = ET.fromstring(xml_str)
    ns = {"cap": CAP_NAMESPACE}
    info_blocks = root.findall("cap:info", ns)
    assert len(info_blocks) == len(SAMPLE_ALERTS)


def test_build_cap_xml_severity_reflects_probability():
    xml_str = build_cap_xml(SAMPLE_ALERTS)
    root = ET.fromstring(xml_str)
    ns = {"cap": CAP_NAMESPACE}
    severities = [info.find("cap:severity", ns).text for info in root.findall("cap:info", ns)]
    assert severities[0] == "Severe"  # probability 0.82
    assert severities[1] == "Minor"  # probability 0.45


def test_build_cap_xml_empty_alerts_still_valid():
    xml_str = build_cap_xml([])
    root = ET.fromstring(xml_str)
    ns = {"cap": CAP_NAMESPACE}
    assert root.find("cap:identifier", ns) is not None
    assert len(root.findall("cap:info", ns)) == 0
