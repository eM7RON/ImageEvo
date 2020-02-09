from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QUrl
import sys


class DirectoryExplorer(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 file dialogs - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.openFileNameDialog()
        self.openFileNamesDialog()
        self.saveFileDialog()
        #self.openFolderDialog()
        
        self.show()
    
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            print(fileName)
    
    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","All Files (*);;Python Files (*.py)", options=options)
        if files:
            print(files)
    
    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            print(fileName)

    def openFolderDialog(self):
        #file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        options = QFileDialog.Options()
        #options |= QFileDialog.DontUseNativeDialog
        dialog = QFileDialog(self, 'Audio Files', '', 'All Files (*)', options=options)
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setSidebarUrls([QUrl.fromLocalFile(place)])
        #file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if dialog.exec_() == QDialog.Accepted:
            self._audio_file = dialog.selectedFiles()[0]

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DirectoryExplorer()
    sys.exit(app.exec_())
