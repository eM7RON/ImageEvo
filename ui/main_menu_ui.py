# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_menu_ui.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainMenu(object):
    def setupUi(self, MainMenu):
        MainMenu.setObjectName("MainMenu")
        MainMenu.resize(357, 412)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainMenu.sizePolicy().hasHeightForWidth())
        MainMenu.setSizePolicy(sizePolicy)
        MainMenu.setMinimumSize(QtCore.QSize(357, 412))
        MainMenu.setMaximumSize(QtCore.QSize(357, 412))
        MainMenu.setSizeIncrement(QtCore.QSize(0, 0))
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(MainMenu)
        self.verticalLayout_2.setSpacing(5)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(MainMenu)
        self.label.setMinimumSize(QtCore.QSize(100, 27))
        self.label.setMaximumSize(QtCore.QSize(100, 27))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.img_widget = QtWidgets.QLabel(MainMenu)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img_widget.sizePolicy().hasHeightForWidth())
        self.img_widget.setSizePolicy(sizePolicy)
        self.img_widget.setMinimumSize(QtCore.QSize(150, 150))
        self.img_widget.setMaximumSize(QtCore.QSize(150, 150))
        self.img_widget.setText("")
        self.img_widget.setPixmap(QtGui.QPixmap("../img/gfx/alg_img.svg"))
        self.img_widget.setScaledContents(True)
        self.img_widget.setAlignment(QtCore.Qt.AlignCenter)
        self.img_widget.setObjectName("img_widget")
        self.horizontalLayout_2.addWidget(self.img_widget)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gpso_button = QtWidgets.QPushButton(MainMenu)
        self.gpso_button.setMinimumSize(QtCore.QSize(0, 27))
        self.gpso_button.setMaximumSize(QtCore.QSize(16777215, 27))
        self.gpso_button.setObjectName("gpso_button")
        self.verticalLayout.addWidget(self.gpso_button)
        self.load_button = QtWidgets.QPushButton(MainMenu)
        self.load_button.setMinimumSize(QtCore.QSize(0, 27))
        self.load_button.setMaximumSize(QtCore.QSize(16777215, 27))
        self.load_button.setObjectName("load_button")
        self.verticalLayout.addWidget(self.load_button)
        self.image_editor_button = QtWidgets.QPushButton(MainMenu)
        self.image_editor_button.setMinimumSize(QtCore.QSize(0, 27))
        self.image_editor_button.setMaximumSize(QtCore.QSize(16777215, 27))
        self.image_editor_button.setObjectName("image_editor_button")
        self.verticalLayout.addWidget(self.image_editor_button)
        self.video_maker_button = QtWidgets.QPushButton(MainMenu)
        self.video_maker_button.setMinimumSize(QtCore.QSize(0, 27))
        self.video_maker_button.setMaximumSize(QtCore.QSize(16777215, 27))
        self.video_maker_button.setObjectName("video_maker_button")
        self.verticalLayout.addWidget(self.video_maker_button)
        self.about_button = QtWidgets.QPushButton(MainMenu)
        self.about_button.setMinimumSize(QtCore.QSize(0, 27))
        self.about_button.setMaximumSize(QtCore.QSize(16777215, 27))
        self.about_button.setObjectName("about_button")
        self.verticalLayout.addWidget(self.about_button)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(MainMenu)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.exit_button = QtWidgets.QPushButton(MainMenu)
        self.exit_button.setMinimumSize(QtCore.QSize(80, 27))
        self.exit_button.setMaximumSize(QtCore.QSize(80, 27))
        self.exit_button.setObjectName("exit_button")
        self.horizontalLayout.addWidget(self.exit_button)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(MainMenu)
        QtCore.QMetaObject.connectSlotsByName(MainMenu)

    def retranslateUi(self, MainMenu):
        _translate = QtCore.QCoreApplication.translate
        MainMenu.setWindowTitle(_translate("MainMenu", "𝕀mage𝔼vo"))
        self.label.setText(_translate("MainMenu", "<html><head/><body><p><span style=\" font-weight:600;\">Main Menu</span></p></body></html>"))
        self.gpso_button.setText(_translate("MainMenu", "Geometric PSO"))
        self.load_button.setText(_translate("MainMenu", "Load"))
        self.image_editor_button.setText(_translate("MainMenu", "Image Editor"))
        self.video_maker_button.setText(_translate("MainMenu", "Video Maker"))
        self.about_button.setText(_translate("MainMenu", "About"))
        self.label_2.setText(_translate("MainMenu", "<html><head/><body><p><span style=\" font-size:6pt;\">eM7RON </span></p><p><span style=\" font-size:6pt;\">16 Mar 2020 v0.7</span></p></body></html>"))
        self.exit_button.setText(_translate("MainMenu", "Exit"))
