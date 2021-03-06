# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'progress_bar_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ProgressBar(object):
    def setupUi(self, ProgressBar):
        ProgressBar.setObjectName("ProgressBar")
        ProgressBar.resize(336, 210)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(ProgressBar.sizePolicy().hasHeightForWidth())
        ProgressBar.setSizePolicy(sizePolicy)
        ProgressBar.setMinimumSize(QtCore.QSize(336, 210))
        ProgressBar.setMaximumSize(QtCore.QSize(336, 210))
        self.verticalLayout = QtWidgets.QVBoxLayout(ProgressBar)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.line_3 = QtWidgets.QFrame(ProgressBar)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.horizontalLayout.addWidget(self.line_3)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.line_2 = QtWidgets.QFrame(ProgressBar)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_2.addWidget(self.line_2)
        self.horizontalLayout_31 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_31.setObjectName("horizontalLayout_31")
        self.progress_bar_label = QtWidgets.QLabel(ProgressBar)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progress_bar_label.sizePolicy().hasHeightForWidth())
        self.progress_bar_label.setSizePolicy(sizePolicy)
        self.progress_bar_label.setText("")
        self.progress_bar_label.setObjectName("progress_bar_label")
        self.horizontalLayout_31.addWidget(self.progress_bar_label)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_31.addItem(spacerItem)
        self.verticalLayout_2.addLayout(self.horizontalLayout_31)
        self.progress_bar = QtWidgets.QProgressBar(ProgressBar)
        self.progress_bar.setProperty("value", 0)
        self.progress_bar.setObjectName("progress_bar")
        self.verticalLayout_2.addWidget(self.progress_bar)
        self.horizontalLayout_59 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_59.setObjectName("horizontalLayout_59")
        self.done_label = QtWidgets.QLabel(ProgressBar)
        self.done_label.setObjectName("done_label")
        self.horizontalLayout_59.addWidget(self.done_label)
        self.verticalLayout_2.addLayout(self.horizontalLayout_59)
        self.horizontalLayout_68 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_68.setObjectName("horizontalLayout_68")
        self.total_label = QtWidgets.QLabel(ProgressBar)
        self.total_label.setObjectName("total_label")
        self.horizontalLayout_68.addWidget(self.total_label)
        self.verticalLayout_2.addLayout(self.horizontalLayout_68)
        self.line = QtWidgets.QFrame(ProgressBar)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_2.addWidget(self.line)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.line_4 = QtWidgets.QFrame(ProgressBar)
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.horizontalLayout.addWidget(self.line_4)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(ProgressBar)
        QtCore.QMetaObject.connectSlotsByName(ProgressBar)

    def retranslateUi(self, ProgressBar):
        _translate = QtCore.QCoreApplication.translate
        ProgressBar.setWindowTitle(_translate("ProgressBar", "VideoMaker"))
        self.progress_bar_label.setToolTip(_translate("ProgressBar", "The input file path (supported types: jpg, jpeg, jpe, jfif, png)"))
        self.done_label.setText(_translate("ProgressBar", "Done:"))
        self.total_label.setText(_translate("ProgressBar", "Total:"))

