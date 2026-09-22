"""Test PySide6 GUI controller and signals."""

from pathlib import Path
import pytest
from PySide6.QtWidgets import QApplication

from main import ACPApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_ui_initialization(qapp):
    base_dir = Path(__file__).resolve().parent.parent
    ui_path = base_dir / "ui" / "demo.ui"
    style_path = base_dir / "ui" / "style.qss"

    controller = ACPApplication(ui_path, style_path)
    assert controller.window is not None
    assert controller.window.windowTitle().startswith("Enterprise ABAC Policy Studio")


def test_ui_sample_selection_and_analysis(qapp):
    base_dir = Path(__file__).resolve().parent.parent
    ui_path = base_dir / "ui" / "demo.ui"
    style_path = base_dir / "ui" / "style.qss"

    controller = ACPApplication(ui_path, style_path)
    
    # Select sample
    controller.window.comboSamples.setCurrentIndex(5)  # "Bác sĩ có thể cập nhật hồ sơ bệnh viện."
    assert "Bác sĩ" in controller.window.txtACP_sentence.toPlainText()

    # Analyze
    controller.analyze_sentence()
    assert controller.window.txtSubject.text() == "Bác sĩ"
    assert controller.window.txtAction.text() == "cập nhật"
    assert controller.window.txtObject.text() == "hồ sơ bệnh viện"

    # Generate XACML
    controller.generate_xacml()
    xacml = controller.window.txtXACML.toPlainText()
    assert "<Policy" in xacml
    assert "Bác sĩ" in xacml

    # Clear
    controller.clear_all()
    assert controller.window.txtACP_sentence.toPlainText() == ""
    assert controller.window.txtSubject.text() == ""
