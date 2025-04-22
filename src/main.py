# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import sys
from PyQt5.QtWidgets import QApplication
from src.gui.main_window import FitsViewer

def main():
    app = QApplication(sys.argv)
    viewer = FitsViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
