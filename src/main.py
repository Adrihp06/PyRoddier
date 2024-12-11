from PyQt5.QtWidgets import QApplication
from gui import FitsViewer
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = FitsViewer()
    viewer.show()
    sys.exit(app.exec_())
