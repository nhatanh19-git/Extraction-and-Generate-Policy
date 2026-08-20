"""Desktop entry point for the Vietnamese ACP extraction application."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import QFile
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtUiTools import QUiLoader

from backend.extractor import AttributeExtractor, ExtractionResult
from backend.nlp_engine import NLPEngine
from backend.policy_generator import generate_policy
from backend.xacml_generator import generate_xacml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ACPApplication:
	"""Connect the existing Qt Designer UI to the backend services."""

	def __init__(self, ui_path: Path) -> None:
		self._loader = QUiLoader()
		ui_file = QFile(str(ui_path))
		if not ui_file.open(QFile.OpenModeFlag.ReadOnly):
			raise FileNotFoundError(f"Không thể mở file UI: {ui_path}")
		try:
			self.window = self._loader.load(ui_file)
		finally:
			ui_file.close()
		if self.window is None:
			raise RuntimeError("Không thể tải giao diện Qt")

		self.nlp_engine = NLPEngine()
		self.extractor = AttributeExtractor()
		self.extraction: ExtractionResult | None = None
		self.policy = None

		self.window.buttonAnalyst.clicked.connect(self.analyze_sentence)
		self.window.buttonGeneratePolicy.clicked.connect(self.generate_policy)
		self.window.buttonGenerateXACML.clicked.connect(self.generate_xacml)
		self.window.buttonGenerate_2.clicked.connect(self.generate_policy)

	def analyze_sentence(self) -> None:
		sentence = self.window.txtACP_sentence.toPlainText().strip()
		if not sentence:
			self._show_error("Vui lòng nhập câu chính sách ACP.")
			return

		try:
			document = self.nlp_engine.process(sentence)
			self.extraction = self.extractor.extract(document)
			self.policy = None
			self.window.txtSubject.setText(self._attribute_text("Subject", self.extraction.subject))
			self.window.txtAction.setText(self._attribute_text("Action", self.extraction.action))
			self.window.txtObject.setText(self._attribute_text("Object", self.extraction.object))
			missing = self._missing_attributes(self.extraction)
			if missing:
				self._show_error(f"Không thể xác định {', '.join(missing)}.")
			else:
				self.window.statusbar.showMessage("Phân tích ACP thành công.")
		except Exception as error:
			logger.exception("ACP analysis failed")
			self._show_error(f"Không thể phân tích câu ACP: {error}")

	def generate_policy(self) -> None:
		if self.extraction is None:
			self._show_error("Vui lòng phân tích câu ACP trước.")
			return
		try:
			self.policy = generate_policy(self.extraction)
			self.window.statusbar.showMessage("ABAC Policy đã được tạo với Effect = Permit.")
		except ValueError as error:
			self._show_error(str(error))

	def generate_xacml(self) -> None:
		if self.policy is None:
			if self.extraction is None:
				self._show_error("Vui lòng phân tích câu ACP trước.")
				return
			try:
				self.policy = generate_policy(self.extraction)
			except ValueError as error:
				self._show_error(str(error))
				return
		try:
			self.window.txtXACML.setPlainText(generate_xacml(self.policy))
			self.window.statusbar.showMessage("XACML Policy đã được tạo.")
		except Exception as error:
			logger.exception("XACML generation failed")
			self._show_error(f"Không thể tạo XACML: {error}")

	@staticmethod
	def _attribute_text(label: str, attribute: object | None) -> str:
		if attribute is None:
			return f"Không thể xác định {label}."
		return attribute.text

	@staticmethod
	def _missing_attributes(extraction: ExtractionResult) -> list[str]:
		return [
			label
			for label, attribute in (("Subject", extraction.subject), ("Action", extraction.action), ("Object", extraction.object))
			if attribute is None or not attribute.text.strip()
		]

	def _show_error(self, message: str) -> None:
		self.window.statusbar.showMessage(message)
		QMessageBox.warning(self.window, "Lỗi", message)


def main() -> int:
	"""Load the provided .ui file and start the Qt event loop."""
	application = QApplication(sys.argv)
	ui_path = Path(__file__).resolve().parent / "ui" / "demo.ui"
	controller = ACPApplication(ui_path)
	controller.window.show()
	return application.exec()


if __name__ == "__main__":
	raise SystemExit(main())
