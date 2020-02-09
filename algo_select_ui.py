# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'algo_select_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_AlgoSelectWindow(object):
    def setupUi(self, AlgoSelectWindow):
        AlgoSelectWindow.setObjectName("AlgoSelectWindow")
        AlgoSelectWindow.resize(342, 272)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(AlgoSelectWindow)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(AlgoSelectWindow)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.hc_button = QtWidgets.QPushButton(AlgoSelectWindow)
        self.hc_button.setObjectName("hc_button")
        self.verticalLayout.addWidget(self.hc_button)
        self.ga_button = QtWidgets.QPushButton(AlgoSelectWindow)
        self.ga_button.setObjectName("ga_button")
        self.verticalLayout.addWidget(self.ga_button)
        self.gpso_button = QtWidgets.QPushButton(AlgoSelectWindow)
        self.gpso_button.setObjectName("gpso_button")
        self.verticalLayout.addWidget(self.gpso_button)
        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(AlgoSelectWindow)
        QtCore.QMetaObject.connectSlotsByName(AlgoSelectWindow)

    def retranslateUi(self, AlgoSelectWindow):
        _translate = QtCore.QCoreApplication.translate
        AlgoSelectWindow.setWindowTitle(_translate("AlgoSelectWindow", "ImageEvo"))
        self.label.setText(_translate("AlgoSelectWindow", "Select an algorithm:"))
        self.hc_button.setText(_translate("AlgoSelectWindow", "Hill Climber"))
        self.ga_button.setText(_translate("AlgoSelectWindow", "Genetic Algorithm"))
        self.gpso_button.setText(_translate("AlgoSelectWindow", "Geometric PSO"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    AlgoSelectWindow = QtWidgets.QWidget()
    ui = Ui_AlgoSelectWindow()
    ui.setupUi(AlgoSelectWindow)
    AlgoSelectWindow.show()
    sys.exit(app.exec_())

