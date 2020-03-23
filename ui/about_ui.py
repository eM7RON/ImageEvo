# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'about_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_AboutWindow(object):
    def setupUi(self, AboutWindow):
        AboutWindow.setObjectName("AboutWindow")
        AboutWindow.resize(704, 360)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(AboutWindow.sizePolicy().hasHeightForWidth())
        AboutWindow.setSizePolicy(sizePolicy)
        AboutWindow.setMinimumSize(QtCore.QSize(704, 360))
        AboutWindow.setMaximumSize(QtCore.QSize(704, 360))
        self.horizontalLayout = QtWidgets.QHBoxLayout(AboutWindow)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.textEdit = QtWidgets.QTextEdit(AboutWindow)
        self.textEdit.setReadOnly(True)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout.addWidget(self.textEdit)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.back_button = QtWidgets.QPushButton(AboutWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.back_button.sizePolicy().hasHeightForWidth())
        self.back_button.setSizePolicy(sizePolicy)
        self.back_button.setMinimumSize(QtCore.QSize(80, 27))
        self.back_button.setMaximumSize(QtCore.QSize(80, 27))
        self.back_button.setSizeIncrement(QtCore.QSize(0, 0))
        self.back_button.setObjectName("back_button")
        self.horizontalLayout_4.addWidget(self.back_button)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout.addLayout(self.verticalLayout)

        self.retranslateUi(AboutWindow)
        QtCore.QMetaObject.connectSlotsByName(AboutWindow)

    def retranslateUi(self, AboutWindow):
        _translate = QtCore.QCoreApplication.translate
        AboutWindow.setWindowTitle(_translate("AboutWindow", "𝕀mage𝔼vo"))
        self.textEdit.setHtml(_translate("AboutWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">About ImageEvo:</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:7pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:7pt;\">This project is intended as a bit of fun and I welcome anybody downloading and/or modifying/adapting any part of it. It started as a uni project whereby I had to adapt an algorithm developed by one of the lecturers to a new novel task. I chose to adapt Geometric Particle Swarm Optimization for evolving images. </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:7pt;\">It is written in Python 3 and makes extensive use of PyQt5, numpy, pillow, aggdraw and cv2 libraries.</span><span style=\" font-size:8pt;\"><br /></span><span style=\" font-size:7pt;\">Currently there is a only a version of GSPO which works but I intend to add other algorithms in the future.</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">References:</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:7pt;\">Moraglio, A., Chio, C.D., Toggelius, J., and Poli, R., 2008. Geometric Particle Swarm Optimization. Journal of Artiﬁcial</span><span style=\" font-size:8pt;\"> </span><span style=\" font-size:7pt;\">Evolution and Applications. DOI= </span><a href=\"https://www.youtube.com/redirect?q=http%3A%2F%2Fdx.doi.org%2Fdoi%3A10.1155%2F2008%2F143624&amp;redir_token=0V26sT4JFjAT3OTN7-yaB-eYW6p8MTU4MzU3MjI5N0AxNTgzNDg1ODk3&amp;event=video_description&amp;v=YbGKnut8iy8\"><span style=\" font-size:7pt; text-decoration: underline; color:#0000ff;\">http://dx.doi.org/doi:10.1155/2008/143624</span></a></p></body></html>"))
        self.back_button.setText(_translate("AboutWindow", "Back"))

