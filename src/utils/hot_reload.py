# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import os
import sys
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

class GUIHotReloader:
    def __init__(self, app, main_window_class):
        self.app = app
        self.main_window_class = main_window_class
        self.window = None
        self.last_modified = {}
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_for_changes)
        self.timer.start(1000)  # Check every second

    def start(self):
        """Start the application with hot reloading enabled."""
        self.window = self.main_window_class()
        self.window.show()
        self.update_last_modified()
        return self.app.exec_()

    def check_for_changes(self):
        """Check if any Python files have been modified."""
        current_modified = {}
        for root, _, files in os.walk('src'):
            for file in files:
                if file.endswith('.py'):
                    path = os.path.join(root, file)
                    current_modified[path] = os.path.getmtime(path)

        # Check if any files have been modified
        for path, mtime in current_modified.items():
            if path not in self.last_modified or mtime > self.last_modified[path]:
                print(f"File {path} has been modified. Reloading...")
                self.reload_application()
                break

        self.last_modified = current_modified

    def reload_application(self):
        """Reload the application by restarting the Python process."""
        python = sys.executable
        os.execl(python, python, *sys.argv)

    def update_last_modified(self):
        """Update the last modified times of all Python files."""
        for root, _, files in os.walk('src'):
            for file in files:
                if file.endswith('.py'):
                    path = os.path.join(root, file)
                    self.last_modified[path] = os.path.getmtime(path)