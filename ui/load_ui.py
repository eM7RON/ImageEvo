# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'load_ui.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_LoadWindow(object):
    def setupUi(self, LoadWindow):
        LoadWindow.setObjectName("LoadWindow")
        LoadWindow.resize(357, 245)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(LoadWindow.sizePolicy().hasHeightForWidth())
        LoadWindow.setSizePolicy(sizePolicy)
        LoadWindow.setMinimumSize(QtCore.QSize(357, 245))
        LoadWindow.setMaximumSize(QtCore.QSize(357, 245))
        self.verticalLayout = QtWidgets.QVBoxLayout(LoadWindow)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.line_3 = QtWidgets.QFrame(LoadWindow)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.horizontalLayout.addWidget(self.line_3)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.line_2 = QtWidgets.QFrame(LoadWindow)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_2.addWidget(self.line_2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(LoadWindow)
        self.label.setMinimumSize(QtCore.QSize(100, 27))
        self.label.setMaximumSize(QtCore.QSize(100, 27))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_2.addItem(spacerItem1)
        self.horizontalLayout_31 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_31.setObjectName("horizontalLayout_31")
        self.input_folder_label_2 = QtWidgets.QLabel(LoadWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.input_folder_label_2.sizePolicy().hasHeightForWidth())
        self.input_folder_label_2.setSizePolicy(sizePolicy)
        self.input_folder_label_2.setObjectName("input_folder_label_2")
        self.horizontalLayout_31.addWidget(self.input_folder_label_2)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_31.addItem(spacerItem2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_31)
        self.horizontalLayout_30 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_30.setObjectName("horizontalLayout_30")
        self.progress_file_line_edit = QtWidgets.QLineEdit(LoadWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progress_file_line_edit.sizePolicy().hasHeightForWidth())
        self.progress_file_line_edit.setSizePolicy(sizePolicy)
        self.progress_file_line_edit.setMinimumSize(QtCore.QSize(184, 25))
        self.progress_file_line_edit.setMaximumSize(QtCore.QSize(184, 100))
        self.progress_file_line_edit.setSizeIncrement(QtCore.QSize(0, 0))
        self.progress_file_line_edit.setBaseSize(QtCore.QSize(0, 0))
        self.progress_file_line_edit.setText("")
        self.progress_file_line_edit.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.progress_file_line_edit.setObjectName("progress_file_line_edit")
        self.horizontalLayout_30.addWidget(self.progress_file_line_edit)
        self.progress_file_browser = QtWidgets.QToolButton(LoadWindow)
        self.progress_file_browser.setMinimumSize(QtCore.QSize(65, 27))
        self.progress_file_browser.setMaximumSize(QtCore.QSize(65, 27))
        self.progress_file_browser.setObjectName("progress_file_browser")
        self.horizontalLayout_30.addWidget(self.progress_file_browser)
        self.verticalLayout_2.addLayout(self.horizontalLayout_30)
        self.horizontalLayout_60 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_60.setObjectName("horizontalLayout_60")
        self.output_folder_label_3 = QtWidgets.QLabel(LoadWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.output_folder_label_3.sizePolicy().hasHeightForWidth())
        self.output_folder_label_3.setSizePolicy(sizePolicy)
        self.output_folder_label_3.setObjectName("output_folder_label_3")
        self.horizontalLayout_60.addWidget(self.output_folder_label_3)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_60.addItem(spacerItem3)
        self.verticalLayout_2.addLayout(self.horizontalLayout_60)
        self.horizontalLayout_59 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_59.setObjectName("horizontalLayout_59")
        self.output_folder_line_edit = QtWidgets.QLineEdit(LoadWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.output_folder_line_edit.sizePolicy().hasHeightForWidth())
        self.output_folder_line_edit.setSizePolicy(sizePolicy)
        self.output_folder_line_edit.setMinimumSize(QtCore.QSize(184, 25))
        self.output_folder_line_edit.setMaximumSize(QtCore.QSize(184, 25))
        self.output_folder_line_edit.setText("")
        self.output_folder_line_edit.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.output_folder_line_edit.setObjectName("output_folder_line_edit")
        self.horizontalLayout_59.addWidget(self.output_folder_line_edit)
        self.output_folder_browser = QtWidgets.QToolButton(LoadWindow)
        self.output_folder_browser.setMinimumSize(QtCore.QSize(65, 27))
        self.output_folder_browser.setMaximumSize(QtCore.QSize(65, 27))
        self.output_folder_browser.setObjectName("output_folder_browser")
        self.horizontalLayout_59.addWidget(self.output_folder_browser)
        self.verticalLayout_2.addLayout(self.horizontalLayout_59)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem4)
        self.line_5 = QtWidgets.QFrame(LoadWindow)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.verticalLayout_2.addWidget(self.line_5)
        self.horizontalLayout_68 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_68.setObjectName("horizontalLayout_68")
        self.run_button = QtWidgets.QPushButton(LoadWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.run_button.sizePolicy().hasHeightForWidth())
        self.run_button.setSizePolicy(sizePolicy)
        self.run_button.setMinimumSize(QtCore.QSize(80, 27))
        self.run_button.setMaximumSize(QtCore.QSize(80, 27))
        self.run_button.setObjectName("run_button")
        self.horizontalLayout_68.addWidget(self.run_button)
        self.back_button = QtWidgets.QPushButton(LoadWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.back_button.sizePolicy().hasHeightForWidth())
        self.back_button.setSizePolicy(sizePolicy)
        self.back_button.setMinimumSize(QtCore.QSize(80, 27))
        self.back_button.setMaximumSize(QtCore.QSize(80, 27))
        self.back_button.setObjectName("back_button")
        self.horizontalLayout_68.addWidget(self.back_button)
        self.verticalLayout_2.addLayout(self.horizontalLayout_68)
        self.line = QtWidgets.QFrame(LoadWindow)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_2.addWidget(self.line)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.line_4 = QtWidgets.QFrame(LoadWindow)
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.horizontalLayout.addWidget(self.line_4)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(LoadWindow)
        QtCore.QMetaObject.connectSlotsByName(LoadWindow)

    def retranslateUi(self, LoadWindow):
        _translate = QtCore.QCoreApplication.translate
        LoadWindow.setWindowTitle(_translate("LoadWindow", "𝕀mage𝔼vo"))
        self.label.setText(_translate("LoadWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Load previous:</span></p></body></html>"))
        self.input_folder_label_2.setToolTip(_translate("LoadWindow", "The input file path (supported types: jpg, jpeg, jpe, jfif, png)"))
        self.input_folder_label_2.setText(_translate("LoadWindow", "Progress file:"))
        self.progress_file_line_edit.setToolTip(_translate("LoadWindow", "The input file path (supported types: jpg, jpeg, jpe, jfif, png)"))
        self.progress_file_browser.setToolTip(_translate("LoadWindow", "Browse for input file (supported types: jpg, jpeg, jpe, jfif, png)"))
        self.progress_file_browser.setText(_translate("LoadWindow", "..."))
        self.output_folder_label_3.setToolTip(_translate("LoadWindow", "The folder in which to output images"))
        self.output_folder_label_3.setText(_translate("LoadWindow", "Output folder:"))
        self.output_folder_line_edit.setToolTip(_translate("LoadWindow", "Images will be output into this folder (Leave blank for no output)"))
        self.output_folder_browser.setToolTip(_translate("LoadWindow", "Browse for output folder"))
        self.output_folder_browser.setText(_translate("LoadWindow", "..."))
        self.run_button.setToolTip(_translate("LoadWindow", "Return to previous screen"))
        self.run_button.setText(_translate("LoadWindow", "Run"))
        self.back_button.setToolTip(_translate("LoadWindow", "Execute the algorithm"))
        self.back_button.setText(_translate("LoadWindow", "Back"))
