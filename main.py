# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QFileDialog, QDesktopWidget, \
                            QDialog, QPushButton, QVBoxLayout, QToolBar, QMessageBox
from PyQt5.QtCore import pyqtSlot, QThread, QUrl, Qt
#from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtGui import QIcon
from algo_select_ui import Ui_AlgoSelectWindow
from gpso_setup_ui import Ui_GpsoSetupWindow
from hillclimber_setup_ui import Ui_HillClimberSetupWindow
from svg_ui import Ui_SvgWidget
from PIL import Image

from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

import matplotlib.pyplot as plt
import numpy as np
import sys, os

plt.style.use('dark_background')
# # Handle high resolution displays:
# if hasattr(Qt, 'AA_EnableHighDpiScaling'):
#     QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

class AlgoSelectWindow(QWidget, Ui_AlgoSelectWindow):

    def __init__(self, parent=None):
        super(AlgoSelectWindow, self).__init__(parent=parent)
        self.setupUi(self)
        dir_ = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QIcon(dir_ + os.path.sep + 'icon.png'))
        self.gpso_button.clicked.connect(self.switch_to_gpso)
        self.hc_button.clicked.connect(self.switch_to_hillclimber)

    @pyqtSlot()
    def switch_to_gpso(self):
        global gpso_window
        self.close()
        gpso_setup_window.show()

    @pyqtSlot()
    def switch_to_hillclimber(self):
        global gpso_window
        self.close()
        hillclimber_setup_window.show() 
    
class GpsoSetupWindow(QWidget, Ui_GpsoSetupWindow):

    valid_color   = "#c4df9b"
    warning_color = "#fff79a"
    invalid_color = "#f6989d" 

    def __init__(self, parent=None):
        super(GpsoSetupWindow, self).__init__(parent=parent)
        self.setupUi(self)
        dir_ = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QIcon(dir_ + os.path.sep + "icon.png"))

        ##############################################################################
        #                             Set Defaults                                   #
        ##############################################################################

        # Dropdown
        self.shape_types = ['circle', 'ellipse', 'square', 'rectangle', 'polygon']
        self.shape_type_dropdown.addItems(self.shape_types)

        self.shape_type_dropdown.setCurrentIndex(self.shape_types.index('polygon'))

        self.bg_color_dropdown.addItems(['black', 'white'])
        self.bg_color_dropdown.setCurrentText('black')

        ##############################################################################
        #                            Set Connections                                 #
        ##############################################################################

        # IO
        self.input_file_browser.clicked.connect(self.getInputFile)
        self.output_folder_browser.clicked.connect(self.getOutputFolder)

        # Dropdown
        self.shape_type_dropdown.currentIndexChanged.connect(self.shapeTypeState)

        # Slider connections
        self.n_pop_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.n_pop_slider, self.n_pop_spinb))
        self.n_vert_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.n_vert_slider, self.n_vert_spinb))
        self.init_shape_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.init_shape_slider, self.init_shape_spinb))
        self.max_shape_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.max_shape_slider, self.max_shape_spinb))

        self.x_bits_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.x_bits_slider, self.x_bits_spinb))
        self.y_bits_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.y_bits_slider, self.y_bits_spinb))
        self.c_bits_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.c_bits_slider, self.c_bits_spinb))

        # Spin box connections
        self.n_pop_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.n_pop_spinb, self.n_pop_slider))
        self.n_vert_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.n_vert_spinb, self.n_vert_slider))
        self.init_shape_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.init_shape_spinb, self.init_shape_slider))
        self.max_shape_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.max_shape_spinb, self.max_shape_slider))

        self.x_bits_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.x_bits_spinb, self.x_bits_slider))
        self.y_bits_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.y_bits_spinb, self.y_bits_slider))
        self.c_bits_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.c_bits_spinb, self.c_bits_slider))

        # Navigation
        self.back_button.clicked.connect(self.back)
        self.run_button.clicked.connect(self.run)

        # Validate text input
        self.input_file_text_input.textChanged.connect(lambda: self.validateInputFile(self.input_file_text_input))
        self.input_file_text_input.textChanged.emit(self.input_file_text_input.text())
        self.input_file_text_input.textChanged.connect(self.validateBitLengths)
        self.input_file_text_input.textChanged.emit(self.input_file_text_input.text())

        self.output_freq_entry.textChanged.connect(self.validateOutputFreq)
        self.output_freq_entry.textChanged.emit(self.output_freq_entry.text())
        self.output_freq_entry.textChanged.connect(self.validateOutputDir)
        self.output_freq_entry.textChanged.emit(self.output_freq_entry.text())
        self.output_folder_text_input.textChanged.connect(self.validateOutputDir)
        self.output_folder_text_input.textChanged.emit(self.output_folder_text_input.text())
        self.output_freq_entry.textChanged.emit(self.output_freq_entry.text())
        self.max_iter_entry.textChanged.connect(self.validateMaxIter)
        self.max_iter_entry.textChanged.emit(self.max_iter_entry.text())

        # Weights
        self.w0_entry.textChanged.connect(lambda: self.validateWeight(self.w0_entry))
        self.w0_entry.textChanged.emit(self.w0_entry.text())
        self.w1_entry.textChanged.connect(lambda: self.validateWeight(self.w1_entry))
        self.w1_entry.textChanged.emit(self.w0_entry.text())
        self.w2_entry.textChanged.connect(lambda: self.validateWeight(self.w2_entry))
        self.w2_entry.textChanged.emit(self.w0_entry.text())

        # Mutation rate
        self.m_bit_flip_entry.textChanged.connect(lambda: self.validateMutationRate(self.m_bit_flip_entry))
        self.m_bit_flip_entry.textChanged.emit(self.m_bit_flip_entry.text())
        self.m_shape_swap_entry.textChanged.connect(lambda: self.validateMutationRate(self.m_shape_swap_entry))
        self.m_shape_swap_entry.textChanged.emit(self.m_shape_swap_entry.text())

        self.inputs =  {k: v for k, v in self.__dict__.items() if k.endswith(("_button", "_entry", "_slider", "_input", '_spinb'))}

    @pyqtSlot()
    def updateSpinBoxSlider(self, sender, receiver):
        receiver.setValue(sender.value())
    
    @pyqtSlot()
    def getInputFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fname, _ = QFileDialog.getOpenFileName(self, "Select an input image", "", "Image files (*.jpg *.jpeg *.jpe *.jfif *.png)", options=options)
        self.input_file_text_input.setText(fname)

    @pyqtSlot()
    def getOutputFolder(self):
        options = QFileDialog.Options() | QFileDialog.DontUseNativeDialog | QFileDialog.ShowDirsOnly
        folder = str(QFileDialog.getExistingDirectory(self, "Select a directory", "", options=options))
        self.output_folder_text_input.setText(folder)

    def setUniVert(self):
        self.n_vert_spinb.setValue(1)
        self.n_vert_spinb.setReadOnly(True)
        self.n_vert_slider.setMinimum(1)
        self.n_vert_slider.setMaximum(1)

    def setDualVert(self):
        self.n_vert_spinb.setValue(2)
        self.n_vert_spinb.setReadOnly(True)
        self.n_vert_slider.setMinimum(2)
        self.n_vert_slider.setMaximum(2)

    def setMultiVert(self):
        self.n_vert_spinb.setReadOnly(False)
        self.n_vert_slider.setMinimum(3)
        self.n_vert_slider.setMaximum(20)

    @pyqtSlot()
    def shapeTypeState(self):
        shape_type = self.shape_type_dropdown.currentText()
        if shape_type in {'circle', 'square'}:
            self.setUniVert()
        elif shape_type in {'ellipse', 'rectangle'}:
            self.setDualVert()
        else:
            self.setMultiVert()

    def isNum(self, x):
        try:
            x = float(x)
        except ValueError:
            return False
        else:
            return round(x)
        

    @pyqtSlot()
    def validateOutputDir(self):
        dir_ = self.output_folder_text_input.text().strip()
        if dir_ and os.path.exists(os.path.normpath(dir_)) \
            or not dir_ and self.output_freq_entry.valid:
            color = self.valid_color
            valid = True
        elif not dir_:
            color = self.warning_color
            valid = True
        else:
            color = self.invalid_color
            valid = False
        self.output_folder_text_input.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
        self.output_folder_text_input.valid = valid

    @pyqtSlot()
    def validateInputFile(self, sender):
        path = sender.text()
        if os.path.exists(os.path.normpath(path)) and path.endswith((".jpg", ".jpeg", ".jpe", ".jfif", ".png")):
            color = self.valid_color
            valid = True
        # elif not sender.text().strip():
        #     color = "#fff79a" # yellow
        #     valid = True
        else:
            color = self.invalid_color
            valid = False
        sender.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
        sender.valid = valid

    @pyqtSlot()
    def validateOutputFreq(self):
        text = self.output_freq_entry.text().strip()
        try:
            num = round(float(text))
        except ValueError:
            color = self.invalid_color
            valid = False
        else:
            u_score = '_' in text
            not_output_dir = self.output_folder_text_input.text().strip()
            if num >= 1 and not u_score \
                or num == 0 and not not_output_dir \
                or not text == '' and not not_output_dir:
                color = self.valid_color
                valid = True
            else:
                color = self.invalid_color
                valid = False
        finally:
            self.output_freq_entry.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
            self.output_freq_entry.valid = valid

    @pyqtSlot()
    def validateWeight(self, sender):
        try:
            text = sender.text()
            num = float(text)
            u_score = '_' in text
            if 0 <= num <= 1 and not u_score:
                color = self.valid_color
                valid = True
            elif num > 0 and not u_score:
                color = self.warning_color
                valid = True
            else:
                color = self.invalid_color
                valid = False
        except ValueError:
            color = self.invalid_color
            valid = False
        sender.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
        sender.valid = valid

    @pyqtSlot()
    def validateMutationRate(self, sender):
        text = sender.text()
        try:
            num = float(text)
        except ValueError:
            color = self.invalid_color
            valid = False
        else:
            u_score = '_' in text
            if 0 <= num <= 1 and not u_score:
                color = self.valid_color
                valid = True
            else:
                color = self.invalid_color
                valid = False
        finally:
            sender.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
            sender.valid = valid

    @pyqtSlot()
    def validateMaxIter(self):
        text = self.max_iter_entry.text().strip()

        if text.lower() in {'inf', 'none'}:
            valid = True
        elif '_' in text:
            valid = False
        else:
            try:
                num = round(float(text))
            except (ValueError, OverflowError) as error:
                valid = False
            else:
                if num >= 0:
                    valid = True
                else:
                    valid = False
        color = self.valid_color if valid else self.invalid_color
        self.max_iter_entry.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
        self.max_iter_entry.valid = valid

    @pyqtSlot()
    def validateBitLengths(self):
        if self.input_file_text_input.valid:
            img = Image.open(os.path.normpath(self.input_file_text_input.text()))
            x, y = img.size
            x, y = self.binMax(x), self.binMax(y)
            self.x_bits_slider.setMaximum(x)
            self.y_bits_slider.setMaximum(y)

    def binMax(self, v):
        u = 1
        while int("1" * u, 2) <= v:
            u += 1
        return u

    def readyCheck(self):
        return all(input_.valid for input_ in self.inputs.values)

    @pyqtSlot()
    def back(self):
        global algo_select_window
        self.close()
        algo_select_window.show()
    
    @pyqtSlot()
    def run(self):
        kwargs = dict(
                      filename    = self.input_file_text_input.text(),
                      directory   = self.output_folder_text_input.text(),
                      max_iter    = 'inf' if self.max_iter_entry.text().strip().lower() in {'inf', 'none'} else round(float(self.max_iter_entry.text())),
                      output_freq = round(float(self.output_freq_entry.text().strip())),
                      bg_color    = self.bg_color_dropdown.currentText(),
                      w           = [
                                     float(self.w0_entry.text()), 
                                     float(self.w1_entry.text()),
                                     float(self.w2_entry.text()),
                                    ],
                      m           = [
                                     float(self.m_bit_flip_entry.text()), 
                                     float(self.m_shape_swap_entry.text()), 
                                     0.002,
                                     1e-2
                                     ],
                      shape_type  = self.shape_type_dropdown.currentText(),
                      n_vert      = self.n_vert_slider.value(),
                      n_pop       = self.n_pop_slider.value(),
                      n_shapes    = self.init_shape_slider.value(),
                      max_shapes  = self.max_shape_slider.value(),
                      x_bits      = self.x_bits_slider.value(),
                      y_bits      = self.y_bits_slider.value(),
                      c_bits      = self.c_bits_slider.value(),
                     )

        global svg_window
        svg_window.init(alg_id='gpso', alg_kwargs=kwargs)
        #svg_window.init_algo(algo)
        self.close()

class HillClimberSetupWindow(QWidget, Ui_HillClimberSetupWindow):

    valid_color   = "#c4df9b"
    warning_color = "#fff79a"
    invalid_color = "#f6989d" 

    def __init__(self, parent=None):
        super(HillClimberSetupWindow, self).__init__(parent=parent)
        self.setupUi(self)
        dir_ = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QIcon(dir_ + os.path.sep + "icon.png"))

        ##############################################################################
        #                             Set Defaults                                   #
        ##############################################################################

        # Dropdown
        self.shape_types = ['circle', 'ellipse', 'square', 'rectangle', 'polygon']
        self.shape_type_dropdown.addItems(self.shape_types)

        self.shape_type_dropdown.setCurrentIndex(self.shape_types.index('polygon'))

        self.bg_color_dropdown.addItems(['black', 'white'])
        self.bg_color_dropdown.setCurrentText('black')

        ##############################################################################
        #                            Set Connections                                 #
        ##############################################################################

        # IO
        self.input_file_browser.clicked.connect(self.getInputFile)
        self.output_folder_browser.clicked.connect(self.getOutputFolder)

        # Dropdown
        self.shape_type_dropdown.currentIndexChanged.connect(self.shapeTypeState)

        # Slider connections
        self.n_pop_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.n_pop_slider, self.n_pop_spinb))
        self.n_vert_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.n_vert_slider, self.n_vert_spinb))
        self.init_shape_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.init_shape_slider, self.init_shape_spinb))
        self.max_shape_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.max_shape_slider, self.max_shape_spinb))

        self.x_bits_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.x_bits_slider, self.x_bits_spinb))
        self.y_bits_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.y_bits_slider, self.y_bits_spinb))
        self.c_bits_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.c_bits_slider, self.c_bits_spinb))

        # Spin box connections
        self.n_pop_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.n_pop_spinb, self.n_pop_slider))
        self.n_vert_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.n_vert_spinb, self.n_vert_slider))
        self.init_shape_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.init_shape_spinb, self.init_shape_slider))
        self.max_shape_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.max_shape_spinb, self.max_shape_slider))

        self.x_bits_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.x_bits_spinb, self.x_bits_slider))
        self.y_bits_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.y_bits_spinb, self.y_bits_slider))
        self.c_bits_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.c_bits_spinb, self.c_bits_slider))

        # Navigation
        self.back_button.clicked.connect(self.back)
        self.run_button.clicked.connect(self.run)

        # Validate text input
        self.input_file_text_input.textChanged.connect(lambda: self.validateInputFile(self.input_file_text_input))
        self.input_file_text_input.textChanged.emit(self.input_file_text_input.text())
        self.input_file_text_input.textChanged.connect(self.validateBitLengths)
        self.input_file_text_input.textChanged.emit(self.input_file_text_input.text())

        self.output_freq_entry.textChanged.connect(self.validateOutputFreq)
        self.output_freq_entry.textChanged.emit(self.output_freq_entry.text())
        self.output_freq_entry.textChanged.connect(self.validateOutputDir)
        self.output_freq_entry.textChanged.emit(self.output_freq_entry.text())
        self.output_folder_text_input.textChanged.connect(self.validateOutputDir)
        self.output_folder_text_input.textChanged.emit(self.output_folder_text_input.text())
        self.output_freq_entry.textChanged.emit(self.output_freq_entry.text())
        self.max_iter_entry.textChanged.connect(self.validateMaxIter)
        self.max_iter_entry.textChanged.emit(self.max_iter_entry.text())

        # Weights
        self.w0_entry.textChanged.connect(lambda: self.validateWeight(self.w0_entry))
        self.w0_entry.textChanged.emit(self.w0_entry.text())
        self.w1_entry.textChanged.connect(lambda: self.validateWeight(self.w1_entry))
        self.w1_entry.textChanged.emit(self.w0_entry.text())
        self.w2_entry.textChanged.connect(lambda: self.validateWeight(self.w2_entry))
        self.w2_entry.textChanged.emit(self.w0_entry.text())

        # Mutation rate
        self.m_bit_flip_entry.textChanged.connect(lambda: self.validateMutationRate(self.m_bit_flip_entry))
        self.m_bit_flip_entry.textChanged.emit(self.m_bit_flip_entry.text())
        self.m_shape_swap_entry.textChanged.connect(lambda: self.validateMutationRate(self.m_shape_swap_entry))
        self.m_shape_swap_entry.textChanged.emit(self.m_shape_swap_entry.text())

        self.inputs =  {k: v for k, v in self.__dict__.items() if k.endswith(("_button", "_entry", "_slider", "_input", '_spinb'))}

    @pyqtSlot()
    def updateSpinBoxSlider(self, sender, receiver):
        receiver.setValue(sender.value())
    
    @pyqtSlot()
    def getInputFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fname, _ = QFileDialog.getOpenFileName(self, "Select an input image", "", "Image files (*.jpg *.jpeg *.jpe *.jfif *.png)", options=options)
        self.input_file_text_input.setText(fname)

    @pyqtSlot()
    def getOutputFolder(self):
        options = QFileDialog.Options() | QFileDialog.DontUseNativeDialog | QFileDialog.ShowDirsOnly
        folder = str(QFileDialog.getExistingDirectory(self, "Select a directory", "", options=options))
        self.output_folder_text_input.setText(folder)

    def setUniVert(self):
        self.n_vert_spinb.setValue(1)
        self.n_vert_spinb.setReadOnly(True)
        self.n_vert_slider.setMinimum(1)
        self.n_vert_slider.setMaximum(1)

    def setDualVert(self):
        self.n_vert_spinb.setValue(2)
        self.n_vert_spinb.setReadOnly(True)
        self.n_vert_slider.setMinimum(2)
        self.n_vert_slider.setMaximum(2)

    def setMultiVert(self):
        self.n_vert_spinb.setReadOnly(False)
        self.n_vert_slider.setMinimum(3)
        self.n_vert_slider.setMaximum(20)

    @pyqtSlot()
    def shapeTypeState(self):
        shape_type = self.shape_type_dropdown.currentText()
        if shape_type in {'circle', 'square'}:
            self.setUniVert()
        elif shape_type in {'ellipse', 'rectangle'}:
            self.setDualVert()
        else:
            self.setMultiVert()

    def isNum(self, x):
        try:
            x = float(x)
        except ValueError:
            return False
        else:
            return round(x)

    @pyqtSlot()
    def validateOutputDir(self):
        dir_ = self.output_folder_text_input.text().strip()
        if dir_ and os.path.exists(os.path.normpath(dir_)) \
            or not dir_ and self.output_freq_entry.valid:
            color = self.valid_color
            valid = True
        elif not dir_:
            color = self.warning_color
            valid = True
        else:
            color = self.invalid_color
            valid = False
        self.output_folder_text_input.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
        self.output_folder_text_input.valid = valid

    @pyqtSlot()
    def validateInputFile(self, sender):
        path = sender.text()
        if os.path.exists(os.path.normpath(path)) and path.endswith((".jpg", ".jpeg", ".jpe", ".jfif", ".png")):
            color = self.valid_color
            valid = True
        # elif not sender.text().strip():
        #     color = "#fff79a" # yellow
        #     valid = True
        else:
            color = self.invalid_color
            valid = False
        sender.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
        sender.valid = valid

    @pyqtSlot()
    def validateOutputFreq(self):
        text = self.output_freq_entry.text().strip()
        try:
            num = round(float(text))
        except ValueError:
            color = self.invalid_color
            valid = False
        else:
            u_score = '_' in text
            not_output_dir = self.output_folder_text_input.text().strip()
            if num >= 1 and not u_score \
                or num == 0 and not not_output_dir \
                or not text == '' and not not_output_dir:
                color = self.valid_color
                valid = True
            else:
                color = self.invalid_color
                valid = False
        finally:
            self.output_freq_entry.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
            self.output_freq_entry.valid = valid

    @pyqtSlot()
    def validateWeight(self, sender):
        try:
            text = sender.text()
            num = float(text)
            u_score = '_' in text
            if 0 <= num <= 1 and not u_score:
                color = self.valid_color
                valid = True
            elif num > 0 and not u_score:
                color = self.warning_color
                valid = True
            else:
                color = self.invalid_color
                valid = False
        except ValueError:
            color = self.invalid_color
            valid = False
        sender.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
        sender.valid = valid

    @pyqtSlot()
    def validateMutationRate(self, sender):
        text = sender.text()
        try:
            num = float(text)
        except ValueError:
            color = self.invalid_color
            valid = False
        else:
            u_score = '_' in text
            if 0 <= num <= 1 and not u_score:
                color = self.valid_color
                valid = True
            else:
                color = self.invalid_color
                valid = False
        finally:
            sender.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
            sender.valid = valid

    @pyqtSlot()
    def validateMaxIter(self):
        text = self.max_iter_entry.text().strip()

        if text.lower() in {'inf', 'none'}:
            valid = True
        elif '_' in text:
            valid = False
        else:
            try:
                num = round(float(text))
            except (ValueError, OverflowError) as error:
                valid = False
            else:
                if num >= 0:
                    valid = True
                else:
                    valid = False
        color = self.valid_color if valid else self.invalid_color
        self.max_iter_entry.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
        self.max_iter_entry.valid = valid

    @pyqtSlot()
    def validateBitLengths(self):
        if self.input_file_text_input.valid:
            img = Image.open(os.path.normpath(self.input_file_text_input.text()))
            x, y = img.size
            x, y = self.binMax(x), self.binMax(y)
            self.x_bits_slider.setMaximum(x)
            self.y_bits_slider.setMaximum(y)

    def binMax(self, v):
        u = 1
        while int("1" * u, 2) <= v:
            u += 1
        return u

    def readyCheck(self):
        return all(input_.valid for input_ in self.inputs.values)

    @pyqtSlot()
    def back(self):
        global algo_select_window
        self.close()
        algo_select_window.show()
    
    @pyqtSlot()
    def run(self):
        kwargs = dict(
                      filename    = self.input_file_text_input.text(),
                      directory   = self.output_folder_text_input.text(),
                      max_iter    = 'inf' if self.max_iter_entry.text().strip().lower() in {'inf', 'none'} else round(float(self.max_iter_entry.text())),
                      output_freq = round(float(self.output_freq_entry.text().strip())),
                      bg_color    = self.bg_color_dropdown.currentText(),
                      w           = [
                                     float(self.w0_entry.text()), 
                                     float(self.w1_entry.text()),
                                     float(self.w2_entry.text()),
                                    ],
                      m           = [
                                     float(self.m_bit_flip_entry.text()), 
                                     float(self.m_shape_swap_entry.text()), 
                                     0.002,
                                     1e-2
                                     ],
                      shape_type  = self.shape_type_dropdown.currentText(),
                      n_vert      = self.n_vert_slider.value(),
                      n_pop       = self.n_pop_slider.value(),
                      n_shapes    = self.init_shape_slider.value(),
                      max_shapes  = self.max_shape_slider.value(),
                      x_bits      = self.x_bits_slider.value(),
                      y_bits      = self.y_bits_slider.value(),
                      c_bits      = self.c_bits_slider.value(),
                     )

        global svg_window
        svg_window.init(alg_id='hillclimber', alg_kwargs=kwargs)
        self.close()


class SvgWindow(QWidget, Ui_SvgWidget):

    def __init__(self, parent=None):
        super(SvgWindow, self).__init__(parent=parent)
        self.setupUi(self)
        dir_ = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QIcon(dir_ + os.path.sep + 'icon.png'))
        shape = QDesktopWidget().screenGeometry()
        self.setGeometry(shape.width() / 2, shape.height() / 2, 400, 400)

    def init(self, alg_id=None, alg_kwargs=None):
        if alg_id == 'gpso':
            ALG = __import__(alg_id).GPSO
        elif alg_id == 'hillclimber':
            ALG = __import__(alg_id).HillClimber
        self.iter_label.setStyleSheet("QLabel { color : white; }")
        self.iter_indicator.setStyleSheet("QLabel { color : white; }")
        self.show()
        self.SvgDisplay.show()
        self.alg_id = alg_id
        self.alg_kwargs = alg_kwargs

        self.alg = ALG(**self.alg_kwargs)
        thread = QThread()
        self.threads = (self.alg, thread)
        self.alg.moveToThread(thread)
        # get progress messages from worker:
        self.alg.update_svg_display.connect(self.updateDisplay)
        self.alg.update_iter_indicator.connect(self.updateIter)
        self.alg.update_n_shapes.connect(self.updateNShapes)
        self.alg.update_performance_metrics.connect(self.PltDisplay.updateFigure)
        self.bst_checkbox.toggled.connect(lambda: self.PltDisplay.toggleVisible('bst'))
        self.wst_checkbox.toggled.connect(lambda: self.PltDisplay.toggleVisible('wst'))
        self.avg_checkbox.toggled.connect(lambda: self.PltDisplay.toggleVisible('avg'))
        self.std_checkbox.toggled.connect(lambda: self.PltDisplay.toggleVisible('std'))
        # get read to start worker:
        # self.sig_start.connect(worker.work)  # needed due to PyCharm debugger bug (!); comment out next line
        thread.started.connect(self.alg.run)
        thread.start()

    # @pyqtSlot(str)
    # def renderSvg(self, svg_str):
    #     svg_bytes = bytearray(svg_str, encoding='utf-8')
    #     self.display.renderer().load(svg_bytes)
    #     self.display.update()
    @pyqtSlot(int)
    def updateNShapes(self, n_shapes):
        self.n_shape_indicator.setText(str(n_shapes))

    @pyqtSlot(int)
    def updateIter(self, i_iter):
        self.iter_indicator.setText(str(i_iter))

    @pyqtSlot(str)
    def updateDisplay(self, svg_str):
        svg_bytes = bytearray(svg_str, encoding='utf-8')
        self.SvgDisplay.renderer().load(svg_bytes)
        self.show()
        self.SvgDisplay.show()

if __name__ == "__main__":
    app = QApplication([])
    if 'Fusion' in __import__('PyQt5').QtWidgets.QStyleFactory.keys():
        app.setStyle('Fusion')
    algo_select_window = AlgoSelectWindow()
    gpso_setup_window = GpsoSetupWindow()
    hillclimber_setup_window = HillClimberSetupWindow()
    svg_window = SvgWindow()
    algo_select_window.show()
    sys.exit(app.exec_())

