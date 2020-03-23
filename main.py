# -*- coding: utf-8 -*-

import sys, os, time, copy, re #, shutil

from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QFileDialog, QDesktopWidget, QStyleFactory, \
                            QDialog, QPushButton, QVBoxLayout, QToolBar, QMessageBox, QProgressBar
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread, QUrl, Qt, QTimer
from PyQt5.QtGui import QIcon, QPixmap, QImage, QPainter
from PyQt5.QtSvg import QSvgWidget, QSvgRenderer
from PIL import Image, ImageFilter
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.pyplot as plt
import numpy as np
import cv2

from ui.load_ui import Ui_LoadWindow
from ui.main_menu_ui import Ui_MainMenu
from ui.gpso_setup_ui import Ui_GpsoSetupWindow
from ui.gpso_display_ui import Ui_GpsoDisplayWindow
from ui.confirmation_prompt_ui import Ui_ConfirmationPrompt
from ui.video_maker_setup_ui import Ui_VideoMakerSetup
from ui.video_maker_prompt_ui import Ui_VideoMakerPrompt
from ui.video_maker_ui import Ui_VideoMaker
from ui.about_ui import Ui_AboutWindow
from ui.image_editor_ui import Ui_ImageEditor
from utils import utils

plt.style.use('dark_background')

if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

VALID_COLOR     = "#c4df9b" # Green
WARNING_COLOR   = "#fff79a" # Yellow
INVALID_COLOR   = "#f6989d" # Red
CWDIR           = os.path.dirname(os.path.realpath(__file__))
ICON_PATH       = os.path.join(CWDIR, 'img', 'dna.svg')
IMG_FORMATS     = (".jpg", ".jpeg", ".jpe", ".jfif", ".png")
IMG_FORMATS_STR = "Image formats (*.jpg *.jpeg *.jpe *.jfif *.png)"

prev_           = None      # Will hold previous screen for navigation back buttons


class VideoMakerPrompt(QWidget, Ui_VideoMakerPrompt):
    '''
    Prompts user to open VideoMaker after the algorithm has finished
    or has been stopped
    '''
    id_   = 'video_maker_prompt'

    def __init__(self, svg_dir=None, parent=None):
        super(VideoMakerPrompt, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowIcon(ICON)
        self.yes_button.clicked.connect(lambda: navigate(self, VideoMakerSetup(svg_dir)))
        self.no_button.clicked.connect(end_program)


class VideoMakerSetup(QWidget, Ui_VideoMakerSetup):

    id_ = 'video_maker_setup'
    container_map = {
        'H264': 'mp4',
        'H265': 'mp4',
        'DIVX': 'avi',
    }

    def __init__(self, svg_dir=None, parent=None):
        super(VideoMakerSetup, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowIcon(ICON)

        # Override self.show to switch back/exit 
        # button text depending on previous screen
        # when self.show is called
        self.__show = copy.copy(self.show)
        self.show   = self._show

        ##############################################################################
        #                      Validated Control Variables                           #
        ##############################################################################

        self.inputs =  {k: v for k, v in self.__dict__.items() if k.endswith(("_entry", "_line_edit"))}

        for v in self.inputs.values():
            v.valid = False

        self.validated_controls = [self.run_button]
        self.close_prompt = ConfirmationPrompt(end_program, window_title='𝕍ideo𝕄aker', message='Quit')

        self.video_codecs = ['h264', 'x264', 'X264', 'h265', 'H264', 'H265', 'DIVX', 'avc1', 'avc3', 'hev1', 'hvc1', 'vc-1', 'drac',
                             'vp09', 'av01', 'ac-3', 'ec-3', 'fLaC', 'tx3g', 'gpmd', 'mp4v', 'Opus', ]
        self.video_codec_dropdown.addItems(self.video_codecs)
        self.video_codec_dropdown.setCurrentIndex(self.video_codecs.index('H264'))
        #self.video_codec_dropdown.activated.connect(self.update_save_as)

        self.containers = ['mp4', 'mov', 'mkv', 'avi', 'divx', 'flv', 'mpg', 'mpeg']
        self.valid_exts = tuple('.' + x for x in self.containers)
        self.container_dropdown.addItems(self.containers)
        self.container_dropdown.setCurrentIndex(self.containers.index('mp4'))
        self.container_dropdown.activated.connect(self.update_save_as)

        self.resolutions = ['720p', '1080p', '1440p', '4K', '8K']
        self.resolution_dropdown.addItems(self.resolutions)
        self.resolution_dropdown.setCurrentIndex(self.resolutions.index('1080p'))

        self.save_as_line_edit.textChanged.connect(lambda: validate_save_name(self, self.valid_exts))
        self.save_as_line_edit.textChanged.emit('')
        self.save_as_file_browser.clicked.connect(lambda: self.save_as(
                                                                       'Where would you like to save?', 
                                                                       'Video ( ' + ''.join([f'*{x} ' for x in self.containers])[: -1] + ')',
                                                                       ))

        self.svg_folder_line_edit.textChanged.connect(self.validate_dir)
        self.svg_folder_line_edit.setText(svg_dir)
        self.svg_folder_line_edit.textChanged.emit(svg_dir)
        self.svg_folder_browser.clicked.connect(lambda: get_dir(self, self.svg_folder_line_edit))

        self.fps_entry.textChanged.connect(lambda: self.validate_value(self.fps_entry))
        self.fps_entry.textChanged.emit(self.fps_entry.text())

        self.run_button.clicked.connect(self.run)

    def _show(self):
        '''
        This method overwrites the built-in self.show method during __init__(). 
        It preserves the orginal self.show functionality but adds the ability to 
        switch between back and exit for one of the buttons depending on the 
        previous screen. It is called with self.show().
        '''
        if prev_.id_ == 'main_menu':
            self.exit_button.setText('Back')
            disconnect(self.exit_button)
            self.exit_button.clicked.connect(lambda: navigate(self, prev_))
        else:
            self.exit_button.setText('Exit')
            disconnect(self.exit_button)
            self.exit_button.clicked.connect(self.close_prompt.show)
        self.__show()

    @pyqtSlot()
    def validate_dir(self):
        dir_ = self.svg_folder_line_edit.text().strip()
        if dir_ and os.path.exists(os.path.normpath(dir_)):
            color = VALID_COLOR
            valid = True
        else:
            color = INVALID_COLOR
            valid = False
        self.svg_folder_line_edit.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
        self.svg_folder_line_edit.valid = valid
        ready_check(self)

    def save_as(self, message='Save as...', exts_str='All Files (*);;Text Files (*.txt)'):
        '''
        Generic dialog for selectring where to save
        '''
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fname, _ = QFileDialog.getSaveFileName(self, message, '', exts_str, options=options)
        if fname:
            if fname.endswith(self.valid_exts):
                self.save_as_line_edit.setText(fname)
                ext   = fname.split('.')[-1]
                self.container_dropdown.setCurrentIndex(self.containers.index(ext))
            else:
                self.save_as_line_edit.setText(utils.replace_extension(fname, self.container_dropdown.currentText()))

    def update_save_as(self):
        if self.save_as_line_edit.valid:
            path = self.save_as_line_edit.text()
            if path.strip():
                dir_, fname = os.path.split(path)
                fname = utils.replace_extension(fname, self.container_dropdown.currentText())
                self.save_as_line_edit.setText(os.path.join(dir_, fname))

    @pyqtSlot()
    def validate_value(self, sender):
        text = sender.text()
        try:
            num = round(float(text))
        except ValueError:
            color = INVALID_COLOR
            valid = False
        else:
            u_score = '_' in text
            if num >= 1 and not u_score:
                color = VALID_COLOR
                valid = True
            else:
                color = INVALID_COLOR
                valid = False
        finally:
            sender.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
            sender.valid = valid
            ready_check(self)

    @pyqtSlot()
    def run(self):
        kwargs = dict(
                      save_name    = self.save_as_line_edit.text(),
                      input_dir    = self.svg_folder_line_edit.text(),
                      video_codec  = self.video_codec_dropdown.currentText(),
                      resolution   = self.resolution_dropdown.currentText(),
                      fps          = round(float(self.fps_entry.text())),             
                      )
        global prev_, video_maker
        prev_ = self
        self.close()
        video_maker = VideoMaker(**kwargs)


class VideoMaker(QWidget, Ui_VideoMaker):
    '''
    Makes videos of the evolving images
    '''
    id_ = 'video_maker'

    resolution_map = {
        '720p' : (1280, 720),
        '1080p': (1920, 1080),
        '1440p': (2560, 1440),
        '4K'   : (3840, 2160),
        '8K'   : (7680, 4320),
    }

    def __init__(self, parent=None, **kwargs):
        super(VideoMaker, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowIcon(ICON)

        self.close_prompt = ConfirmationPrompt(end_program, window_title='𝕍ideo𝕄aker', message='Quit')
        self.back_button.setEnabled(False)
        self.exit_button.setEnabled(False)
        self.back_button.clicked.connect(lambda: navigate(self, prev_))
        self.exit_button.clicked.connect(self.close_prompt.show)

        self.show()

        resolution    = kwargs.get('resolution', '1080p')
        resolution    = self.resolution_map[resolution]
        fps           = kwargs.get('fps', 30)
        video_codec   = kwargs.get('video_codec', 'X264')
        input_dir     = kwargs.get('input_dir', '.\\')
        save_name     = kwargs['save_name']

        self.progress_bar_label.setText('Status: Scanning for SVG images...')
        fnames = sorted(utils.directory_explorer('svg', input_dir), key=utils.natural_order)
        os.chdir(input_dir)
        img_dims = utils.get_svg_dimensions(fnames[0])
        # Attempt to preserve aspect ratio of original image
        width, height = utils.fit_to_screen(resolution, img_dims)
        n        = len(fnames)
        step     = 100. / n
        progress = 0.
        done     = 0
        self.total_value.setText(str(n))
        self.done_value.setText('0')

        svg_widget = QSvgWidget()
        svg_widget.setFixedSize(width, height)
        pixmap = QPixmap(svg_widget.size())
        # X264, .mkv, 60fps, 1080p
        # avc1, .mov  
        # https://github.com/cisco/openh264/releases                      'X264', 'DIVX'
        video_writer = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*video_codec), fps, (width, height))

        self.progress_bar_label.setText('Status: Rendering video...')

        for fn in fnames:
            svg_widget.renderer().load(fn)
            svg_widget.render(pixmap)
            svg_widget.hide()
            svg_widget.close()

            img = pixmap.toImage()
            arr = utils.qimage_to_array(img)[:, :, :3]
            video_writer.write(arr)

            progress += step
            done     += 1
            self.progress_bar.setValue(round(progress))
            self.done_value.setText(str(done))
            QApplication.processEvents()

        cv2.destroyAllWindows()
        video_writer.release()

        self.exit_button.setEnabled(True)
        self.back_button.setEnabled(True)
        self.progress_bar_label.setText('Status: Finished')


class AboutWindow(QWidget, Ui_AboutWindow):
    '''
    A simple explanation of the sofware
    '''
    id_ =  'about_window'

    def __init__(self, parent=None):
        super(AboutWindow, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowIcon(ICON)
        self.back_button.clicked.connect(lambda: navigate(self, prev_))


class MainMenu(QWidget, Ui_MainMenu):
    '''
    The main hub of the software where different algorithms and options can be selected
    '''
    id_ = 'main_menu'

    def __init__(self, parent=None):
        super(MainMenu, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowIcon(ICON)

        self.close_prompt = ConfirmationPrompt(end_program, message='Quit')

        self.load_button.clicked.connect(lambda: navigate(self, load_window))
        self.gpso_button.clicked.connect(lambda: navigate(self, gpso_setup_window))
        self.about_button.clicked.connect(lambda: navigate(self, about_window))
        self.image_editor_button.clicked.connect(lambda: navigate(self, image_editor))
        self.video_maker_button.clicked.connect(lambda: navigate(self, video_maker_setup))
        self.exit_button.clicked.connect(self.close_prompt.show)


class LoadWindow(QWidget, Ui_LoadWindow):
    '''
    Load up a previously saved algorithm
    '''
    id_ = 'load_window'

    def __init__(self, parent=None):
        super(LoadWindow, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowIcon(ICON)

        ##############################################################################
        #                      Validated Control Variables                           #
        ##############################################################################

        self.inputs =  {k: v for k, v in self.__dict__.items() if k.endswith(("_input", "_line_edit"))}

        for v in self.inputs.values():
            v.valid = False

        self.validated_controls = [self.run_button]

        ##############################################################################
        #                            Set Connections                                 #
        ##############################################################################

        # IO
        self.progress_file_browser.clicked.connect(lambda: get_file(self, self.progress_file_line_edit, 
            "Select a previously saved optimizer", "Pickle files (*.pkl)"))
        self.output_folder_browser.clicked.connect(lambda: get_dir(self, self.output_folder_line_edit))

        # Text input
        self.output_folder_line_edit.textChanged.connect(self.validateOutputDir)
        self.output_folder_line_edit.textChanged.emit(self.output_folder_line_edit.text())
        self.progress_file_line_edit.textChanged.connect(self.validateProgressDir)
        self.progress_file_line_edit.textChanged.emit(self.progress_file_line_edit.text())

        # Navigation
        self.back_button.clicked.connect(lambda: navigate(self, prev_))
        self.run_button.clicked.connect(self.run)

    @pyqtSlot()
    def validateProgressDir(self):
        dir_ = self.progress_file_line_edit.text().strip()
        if dir_ and os.path.exists(os.path.normpath(dir_)) and dir_.endswith('.pkl'):
            color = VALID_COLOR
            valid = True
        else:
            color = INVALID_COLOR
            valid = False
        self.progress_file_line_edit.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
        self.progress_file_line_edit.valid = valid
        ready_check(self)

    @pyqtSlot()
    def validateOutputDir(self):
        dir_ = self.output_folder_line_edit.text().strip()
        if dir_ and os.path.exists(os.path.normpath(dir_)):
            color = VALID_COLOR
            valid = True
        elif not dir_:
            color = WARNING_COLOR
            valid = True
        else:
            color = INVALID_COLOR
            valid = False
        self.output_folder_line_edit.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
        self.output_folder_line_edit.valid = valid
        ready_check(self)
    
    @pyqtSlot()
    def run(self):
        kwargs = dict(
                      output_dir   = self.output_folder_line_edit.text(),
                      progress_dir = self.progress_file_line_edit.text(),
                      load_flag    = True,
                     )

        global gpso_display_window
        gpso_display_window = GpsoDisplayWindow()
        gpso_display_window.init(**kwargs)
        self.close()


class ImageEditor(QWidget, Ui_ImageEditor):
    '''
    Setup for the GPSO algorithm
    '''
    id_ = 'gpso_setup_window'
    preset_filters = [
        '', 'blur', 'contour', 'detail', 'edge_enhance', 'edge_enhance_more',
        'emboss', 'sharpen', 
        'smooth', 'smooth_more'
    ]
    filters = [
         '', 'box', 'gaussian', 'median', 'mode', 'min', 'max'
    ]
    filter_map = {
        'box'     : 'BoxBlur',
        'gaussian': 'GaussianBlur',
        'median'  : 'MedianFilter',
        'mode'    : 'ModeFilter',
        'min'     : 'MinFilter',
        'max'     : 'MaxFilter',
    }

    def __init__(self, parent=None):
        super(ImageEditor, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowIcon(ICON)

        self.message_timer = QTimer(self)
        self.message_timer.setInterval(2500)
        self.message_timer.timeout.connect(self.clear_message)

        # For Validating inputs and identifying ready state
        self.inputs =  {k: v for k, v in self.__dict__.items() if k.endswith(("_entry", "_line_edit"))}

        for v in self.inputs.values():
            v.valid = False

        self.validated_controls = [self.save_button, self.apply_button]

        self.input_file_browser.clicked.connect(lambda: get_file (
                                                                 self,  
                                                                 self.input_file_line_edit, 
                                                                 "Select an input image", 
                                                                 IMG_FORMATS_STR,
                                                                 ))
        self.input_file_line_edit.textChanged.connect(lambda: validate_file(self, IMG_FORMATS))
        self.input_file_line_edit.textChanged.connect(self.load_img)
        self.input_file_line_edit.textChanged.emit(self.input_file_line_edit.text())

        self.save_as_browser.clicked.connect(lambda: self.save_as(
                                             exts_str=f'Image (*.{self.input_ext()})'))
        self.save_as_line_edit.textChanged.connect(lambda: validate_save_name(self, self.input_ext()))
        self.save_as_line_edit.textChanged.emit('')
        
        self.size_dropdown.addItems(['64', '128', '256', '512', '1024', '2048', '4096'])
        self.size_dropdown.setCurrentText('256')

        self.kernel_size_entry.textChanged.connect(self.validate_kernel_size)
        self.kernel_size_entry.textChanged.emit('')

        self.filter_dropdown.addItems(self.filters)
        self.filter_dropdown.activated.connect(lambda: self.preset_filter_dropdown.setCurrentText(''))
        self.filter_dropdown.setCurrentText('')
        self.preset_filter_dropdown.addItems(self.preset_filters)
        self.preset_filter_dropdown.activated.connect(lambda: self.filter_dropdown.setCurrentText(''))
        self.preset_filter_dropdown.setCurrentText('')

        self.back_button.clicked.connect(lambda: navigate(self, prev_))
        self.save_button.clicked.connect(self.save_img)
        self.apply_button.clicked.connect(self.apply_changes)

    def load_img(self):
        try:
            self.img = Image.open(os.path.normpath(self.input_file_line_edit.text())).convert('RGBA')
            self.display_img()
        except (FileNotFoundError, AttributeError, PermissionError):
            pass

    def resize_img(self):
        target = int(self.size_dropdown.currentText())
        w, h = utils.fit_to_screen((target, target), self.img.size)
        self.img = self.img.resize((w, h))

    def save_img(self):
        self.img.save(self.save_as_line_edit.text(), 'PNG')
        self.display_message('Saved...')

    def display_img(self):
        size = utils.fit_to_screen((192, 108), self.img.size)
        img  = self.img.resize(size)
        data = img.tobytes()
        img = QImage(data, *img.size, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(img)
        self.img_widget.setPixmap(pixmap)
        self.img_widget.show()

    def apply_changes(self):

        self.resize_img()
        filter_required = False

        if self.preset_filter_dropdown.currentText():
            filter_name = self.preset_filter_dropdown.currentText().upper()
            filter_required = True
        elif self.filter_dropdown.currentText():
            filter_name = self.filter_map[self.filter_dropdown.currentText()]
            filter_required = True
        
        if filter_required:
            filter_ = getattr(ImageFilter, filter_name)

            self.img = Image.filter(filter_(float(self.kernel_size_entry.text())))
        self.display_img()
        self.display_message('Changes applied...')
            

    def input_ext(self):
        return self.input_file_line_edit.text().split('.')[-1]
    
    def save_as(self, message='Save as...', exts_str='All Files (*);;Text Files (*.txt)'):
        '''
        Generic dialog for selectring where to save
        '''
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fname, _ = QFileDialog.getSaveFileName(self, message, '', exts_str, options=options)
        if fname:
            if fname.endswith(IMG_FORMATS):
                self.save_as_line_edit.setText(fname)
            else:
                self.save_as_line_edit.setText(utils.replace_extension(fname, self.input_ext()))

    @pyqtSlot()
    def validate_kernel_size(self):
        text    = self.kernel_size_entry.text()
        kernel1 = self.filter_dropdown.currentText()
        kernel2 = self.preset_filter_dropdown.currentText()
        if not any([text, kernel1, kernel2]):
            color = VALID_COLOR
            valid = True
        elif any([kernel1, kernel2]) and text:
            try:
                num = float(text)
            except ValueError:
                color = INVALID_COLOR
                valid = False
            else:
                u_score = '_' in text
                if num > 0 and not u_score:
                    color = VALID_COLOR
                    valid = True
                else:
                    color = INVALID_COLOR
                    valid = False
        else:
            color = INVALID_COLOR
            valid = False
        self.kernel_size_entry.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
        self.kernel_size_entry.valid = valid
        ready_check(self)

    @pyqtSlot()
    def display_message(self, message):
        self.message_label.setText(message)
        self.message_timer.start()

    @pyqtSlot()
    def clear_message(self):
        self.message_timer.start()
        self.message_label.setText('')

    
class GpsoSetupWindow(QWidget, Ui_GpsoSetupWindow):
    '''
    Setup for the GPSO algorithm
    '''
    id_ = 'gpso_setup_window'

    def __init__(self, parent=None):
        super(GpsoSetupWindow, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowIcon(ICON)

        # For Validating inputs and identifying ready state
        self.inputs =  {k: v for k, v in self.__dict__.items() if k.endswith(("_entry", "_line_edit"))}

        for v in self.inputs.values():
            v.valid = False

        self.validated_controls = [self.run_button]

        ##############################################################################
        #                             Set Defaults                                   #
        ##############################################################################

        # Shape types dropdown
        self.shape_types = ['circle', 'ellipse', 'square', 'rectangle', 'polygon']
        self.shape_type_dropdown.addItems(self.shape_types)

        self.shape_type_dropdown.setCurrentIndex(self.shape_types.index('polygon'))

        self.bg_color_dropdown.addItems(['black', 'white'])
        self.bg_color_dropdown.setCurrentText('black')

        self.shape_management_methods = ['velocity', 'probabilistic', 'periodic']
        self.shape_management_method_dropdown.addItems(self.shape_management_methods)
        self.shape_management_method_dropdown.setCurrentIndex(self.shape_management_methods.index('probabilistic'))

        ##############################################################################
        #                            Set Connections                                 #
        ############################################################################## 

        # I O
        self.input_file_browser.clicked.connect(lambda: get_file (
                                                                 self,  
                                                                 self.input_file_line_edit, 
                                                                 "Select an input image", 
                                                                 IMG_FORMATS_STR,
                                                                 ))
        self.output_folder_browser.clicked.connect(lambda: get_dir(self, self.output_folder_line_edit))
        self.progress_folder_browser.clicked.connect(lambda: get_dir(self, self.progress_folder_line_edit))

        # Shape type dropdown
        self.shape_type_dropdown.currentIndexChanged.connect(self.shape_type_state)

        # Shape increase method dropdown
        self.shape_management_method_dropdown.currentIndexChanged.connect(self.shapeManagementMethodState)
        self.shape_management_method_dropdown.setCurrentIndex(self.shape_management_methods.index('velocity'))

        # Slider connections
        self.n_pop_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.n_pop_slider, self.n_pop_spinb))
        self.n_vert_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.n_vert_slider, self.n_vert_spinb))
        self.init_shape_slider.id = 'init_shape_slider'
        self.max_shape_slider.id  = 'max_shape_slider'
        self.init_shape_slider.valueChanged.connect(lambda: self.update_n_shapespinBoxSlider(self.init_shape_slider, self.init_shape_spinb))
        self.max_shape_slider.valueChanged.connect(lambda: self.update_n_shapespinBoxSlider(self.max_shape_slider, self.max_shape_spinb))

        self.x_bits_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.x_bits_slider, self.x_bits_spinb))
        self.y_bits_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.y_bits_slider, self.y_bits_spinb))
        self.c_bits_slider.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.c_bits_slider, self.c_bits_spinb))

        # Spin box connections
        self.n_pop_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.n_pop_spinb, self.n_pop_slider))
        self.n_vert_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.n_vert_spinb, self.n_vert_slider))
        self.init_shape_spinb.id = 'init_shape_spinb'
        self.max_shape_spinb.id  = 'max_shape_spinb'
        self.init_shape_spinb.valueChanged.connect(lambda: self.update_n_shapespinBoxSlider(self.init_shape_spinb, self.init_shape_slider))
        self.max_shape_spinb.valueChanged.connect(lambda: self.update_n_shapespinBoxSlider(self.max_shape_spinb, self.max_shape_slider))

        self.x_bits_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.x_bits_spinb, self.x_bits_slider))
        self.y_bits_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.y_bits_spinb, self.y_bits_slider))
        self.c_bits_spinb.valueChanged.connect(lambda: self.updateSpinBoxSlider(self.c_bits_spinb, self.c_bits_slider))

        # Navigation
        self.back_button.clicked.connect(lambda: navigate(self, prev_))
        self.run_button.clicked.connect(self.run)

        # Text input
        self.input_file_line_edit.textChanged.connect(lambda: validate_file(self, IMG_FORMATS))
        self.input_file_line_edit.textChanged.emit(self.input_file_line_edit.text())
        self.input_file_line_edit.textChanged.connect(self.validateBitLengths)
        self.input_file_line_edit.textChanged.emit(self.input_file_line_edit.text())

        self.output_freq_entry.textChanged.connect(self.validateOutputFreq)
        self.output_freq_entry.textChanged.emit(self.output_freq_entry.text())
        self.output_freq_entry.textChanged.connect(self.validateOutputDir)
        self.output_freq_entry.textChanged.emit(self.output_freq_entry.text())
        self.output_folder_line_edit.textChanged.connect(self.validateOutputDir)
        self.output_folder_line_edit.textChanged.emit(self.output_folder_line_edit.text())
        self.save_freq_entry.textChanged.connect(self.validateSaveFreq)
        self.save_freq_entry.textChanged.emit(self.save_freq_entry.text())
        self.progress_folder_line_edit.textChanged.connect(self.validateProgressDir)
        self.progress_folder_line_edit.textChanged.emit(self.progress_folder_line_edit.text())

        self.shape_management_value_entry.textChanged.connect(lambda: self.validateShapeManagementValue(self.shape_management_value_entry))
        self.shape_management_value_entry.textChanged.emit(self.shape_management_value_entry.text())

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

    @pyqtSlot()
    def updateSpinBoxSlider(self, sender, receiver):
        receiver.setValue(sender.value())

    @pyqtSlot()
    def update_n_shapespinBoxSlider(self, sender, receiver):
        value = sender.value()
        receiver.setValue(value)
        if sender.id.startswith('init') and value > self.max_shape_slider.value():
            self.max_shape_slider.setValue(value)
            self.max_shape_spinb.setValue(value)
        if sender.id.startswith('max') and value < self.init_shape_slider.value():
            self.init_shape_slider.setValue(value)
            self.init_shape_spinb.setValue(value)

    def set_uni_verts(self):
        self.n_vert_spinb.setValue(1)
        self.n_vert_spinb.setReadOnly(True)
        self.n_vert_slider.setMinimum(1)
        self.n_vert_slider.setMaximum(1)

    def set_dual_verts(self):
        self.n_vert_spinb.setValue(2)
        self.n_vert_spinb.setReadOnly(True)
        self.n_vert_slider.setMinimum(2)
        self.n_vert_slider.setMaximum(2)

    def set_multi_verts(self):
        self.n_vert_spinb.setReadOnly(False)
        self.n_vert_slider.setMinimum(3)
        self.n_vert_slider.setMaximum(20)

    @pyqtSlot()
    def shape_type_state(self):
        shape_type = self.shape_type_dropdown.currentText()
        if shape_type in {'circle', 'square'}:
            self.set_uni_verts()
        elif shape_type in {'ellipse', 'rectangle'}:
            self.set_dual_verts()
        else:
            self.set_multi_verts()

    @pyqtSlot()
    def shapeManagementMethodState(self):
        shape_management_method = self.shape_management_method_dropdown.currentText()
        if shape_management_method == 'probabilistic':
            self.shape_management_value_label.setText('probability')
            self.shape_management_value_entry.setText('1e-3')
        else:
            self.shape_management_value_label.setText('iterations')
            self.shape_management_value_entry.setText('150')

    @pyqtSlot()
    def validateShapeManagementValue(self, sender):
        shape_management_method = self.shape_management_method_dropdown.currentText()
        if shape_management_method == 'probabilistic':
            self.validateWeight(sender)
        else:
            self.validateIterations(sender)
        ready_check(self)   

    @pyqtSlot()
    def validateIterations(self, sender):
        text = sender.text()
        try:
            num = round(float(text))
        except ValueError:
            color = INVALID_COLOR
            valid = False
        else:
            u_score = '_' in text
            if num >= 1 and not u_score:
                color = VALID_COLOR
                valid = True
            else:
                color = INVALID_COLOR
                valid = False
        finally:
            self.shape_management_value_entry.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
            self.shape_management_value_entry.valid = valid
        ready_check(self)

    @pyqtSlot()
    def validateOutputDir(self):
        dir_ = self.output_folder_line_edit.text().strip()
        if dir_ and os.path.exists(os.path.normpath(dir_)) \
            or not dir_ and self.output_freq_entry.valid:
            color = VALID_COLOR
            valid = True
        elif not dir_:
            color = WARNING_COLOR
            valid = True
        else:
            color = INVALID_COLOR
            valid = False
        self.output_folder_line_edit.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
        self.output_folder_line_edit.valid = valid
        ready_check(self)

    @pyqtSlot()
    def validateProgressDir(self):
        dir_ = self.progress_folder_line_edit.text().strip()
        if dir_ and os.path.exists(os.path.normpath(dir_)) \
            or not dir_ and self.save_freq_entry.valid:
            color = VALID_COLOR
            valid = True
        else:
            color = INVALID_COLOR
            valid = False
        self.progress_folder_line_edit.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
        self.progress_folder_line_edit.valid = valid
        ready_check(self)

    @pyqtSlot()
    def validateOutputFreq(self):
        text = self.output_freq_entry.text().strip()
        try:
            num = round(float(text))
        except ValueError:
            color = INVALID_COLOR
            valid = False
        else:
            u_score = '_' in text
            not_output_dir = self.output_folder_line_edit.text().strip()
            if num >= 1 and not u_score \
                or num == 0 and not not_output_dir \
                or not text == '' and not not_output_dir:
                color = VALID_COLOR
                valid = True
            else:
                color = INVALID_COLOR
                valid = False
        finally:
            self.output_freq_entry.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
            self.output_freq_entry.valid = valid
            ready_check(self)

    @pyqtSlot()
    def validateSaveFreq(self):
        text = self.save_freq_entry.text().strip()
        try:
            num = round(float(text))
        except ValueError:
            color = INVALID_COLOR
            valid = False
        else:
            u_score = '_' in text
            if num >= 1 or num == 0:
                color = VALID_COLOR
                valid = True
            else:
                color = INVALID_COLOR
                valid = False
        finally:
            self.save_freq_entry.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
            self.save_freq_entry.valid = valid
            ready_check(self)

    @pyqtSlot()
    def validateWeight(self, sender):
        text = sender.text()
        try:
            num = float(text)
        except ValueError:
            color = INVALID_COLOR
            valid = False
        else:
            u_score = '_' in text
            if 0 <= num <= 1 and not u_score:
                color = VALID_COLOR
                valid = True
            elif num > 0 and not u_score:
                color = WARNING_COLOR
                valid = True
            else:
                color = INVALID_COLOR
                valid = False
        finally:
            sender.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
            sender.valid = valid
            ready_check(self)

    @pyqtSlot()
    def validateMutationRate(self, sender):
        text = sender.text()
        try:
            num = float(text)
        except ValueError:
            color = INVALID_COLOR
            valid = False
        else:
            u_score = '_' in text
            if 0 <= num <= 1 and not u_score:
                color = VALID_COLOR
                valid = True
            else:
                color = INVALID_COLOR
                valid = False
        finally:
            sender.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
            sender.valid = valid
            ready_check(self)

    @pyqtSlot()
    def validateBitLengths(self):
        if self.input_file_line_edit.valid:
            img = Image.open(os.path.normpath(self.input_file_line_edit.text()))
            x, y = img.size
            x, y = utils.bit_max(x), utils.bit_max(y)
            self.x_bits_slider.setMaximum(x)
            self.y_bits_slider.setMaximum(y)
        ready_check(self)
    
    @pyqtSlot()
    def run(self):
        kwargs = dict(
                      img_path      = self.input_file_line_edit.text(),
                      output_dir    = self.output_folder_line_edit.text(),
                      progress_dir  = self.progress_folder_line_edit.text(),
                      output_freq   = round(float(self.output_freq_entry.text().strip())),
                      save_freq     = round(float(self.save_freq_entry.text().strip())),
                      bg_color      = self.bg_color_dropdown.currentText(),
                      w             = [
                                       float(self.w0_entry.text()), 
                                       float(self.w1_entry.text()),
                                       float(self.w2_entry.text()),
                                      ],
                      m             = [
                                       float(self.m_bit_flip_entry.text()),
                                       float(self.m_shape_swap_entry.text()),
                                       ],
                      shape_type    = self.shape_type_dropdown.currentText(),
                      n_vert        = self.n_vert_slider.value(),
                      n_pop         = self.n_pop_slider.value(),
                      n_shape       = self.init_shape_slider.value(),
                      max_shapes    = self.max_shape_slider.value(),
                      x_bits        = self.x_bits_slider.value(),
                      y_bits        = self.y_bits_slider.value(),
                      c_bits        = self.c_bits_slider.value(),
                      oaat_mode     = self.oaat_checkbox.isChecked(),
                      rollback_mode = self.rollback_checkbox.isChecked(),
                      shape_management_func        = self.shape_management_method_dropdown.currentText(),
                      shape_management_probability = round(float(self.shape_management_value_entry.text())),
                      shape_management_delta       = round(float(self.shape_management_value_entry.text())),
                      shape_management_interval    = round(float(self.shape_management_value_entry.text())),                
                     )

        global gpso_display_window
        gpso_display_window = GpsoDisplayWindow()
        gpso_display_window.init(**kwargs)
        self.close()


class ConfirmationPrompt(QWidget, Ui_ConfirmationPrompt):
    '''
    A generic window that can be used for multiple tasks. It prompts the user
    to make sure they want to do whatever they clicked on with "Are you sure?".
    Above this a custom message can be displayed by using the 'message' attribute.
    '''
    id_ = 'confirmation_prompt'

    def __init__(self, yes_action, no_action=None, window_title=None, message=None, parent=None):
        super(ConfirmationPrompt, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowIcon(ICON)
        if window_title is not None:
            self.setWindowTitle(window_title)

        self.message.setText(message if message is not None else '')

        self.yes_button.clicked.connect(yes_action)
        self.no_button.clicked.connect(no_action if no_action is not None else self.close)


class GpsoDisplayWindow(QWidget, Ui_GpsoDisplayWindow):
    '''
    When running, the GPSO algorithm preogress is displayed in this window
    '''
    id_             = 'gpso_disply_window'
    save_pressed    = pyqtSignal()
    pause_pressed   = pyqtSignal()

    def __init__(self, parent=None):
        super(GpsoDisplayWindow, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowIcon(ICON)
        shape = QDesktopWidget().screenGeometry()
        self.setGeometry(shape.width() // 2, shape.height() // 2, 400, 400)

        self.button_map = {
                          'Save state': self.save_pressed,
                          'Pause': self.pause_pressed,
                          'Unpause': self.pause_pressed,
                          }

    def init(self, **kwargs):
        from algorithms.gpso import GPSO
        self.show()
        self.svg_widget.show()
        kwargs['parent_display'] = self

        self.alg = GPSO(**kwargs)
        self.alg_thread = utils.EventLoopThread()
        self.alg.moveToThread(self.alg_thread)

        self.message_timer = QTimer(self)
        self.message_timer.setInterval(2500)
        self.message_timer.timeout.connect(self.clear_message)

        current_size = self.svg_widget.geometry().width(), self.svg_widget.geometry().height()
        scaled_size  = utils.fit_to_screen(current_size, self.alg.img_size)
        self.svg_widget.setFixedSize(*scaled_size)

        # get progress messages from worker:
        self.n_pop_indicator.setText(str(self.alg.n_pop))
        self.n_shape_indicator.setText(str(self.alg.n_shape))

        self.alg.display_signal.connect(self.update_display)
        self.alg.iter_indicators_signal.connect(self.update_iter_indicators)
        self.alg.n_shape_signal.connect(self.update_n_shapes)
        self.alg.fitness_indicator_signal.connect(self.update_fitness_indicator)
        self.alg.performance_metrics_signal.connect(self.MplDisplay.updateFigure)
        self.alg.status_signal.connect(self.display_message)
        self.bst_checkbox.toggled.connect(lambda: self.MplDisplay.toggleVisible('best'))
        self.wst_checkbox.toggled.connect(lambda: self.MplDisplay.toggleVisible('worst'))
        self.avg_checkbox.toggled.connect(lambda: self.MplDisplay.toggleVisible('avg'))
        self.std_checkbox.toggled.connect(lambda: self.MplDisplay.toggleVisible('std'))

        self.close_prompt = ConfirmationPrompt(lambda: navigate_to_video_maker_prompt(self), message='Quit')

        self.pause_button.clicked.connect(lambda: self.execute_action_button(self.pause_button))
        self.save_button.clicked.connect(lambda: self.execute_action_button(self.save_button))
        self.exit_button.clicked.connect(self.close_prompt.show)

        self.alg_thread.started.connect(self.alg.run)
        self.alg_thread.start()

    @pyqtSlot(int)
    def update_n_shapes(self, n_shape):
        self.n_shape_indicator.setText(str(n_shape))

    @pyqtSlot(object)
    def update_iter_indicators(self, iter_data):
        self.iter_indicator.setText(str(iter_data[0]))
        self.isi_indicator.setText(str(iter_data[1]))

    @pyqtSlot(float)
    def update_fitness_indicator(self, distance):
        fitness = 1. / distance
        self.distance_indicator.setText('%.3e' % distance)
        self.fitness_indicator.setText('%.3e' % fitness)

    @pyqtSlot(str)
    def update_display(self, svg_str):
        svg_bytes = bytearray(svg_str, encoding='utf-8')
        self.svg_widget.renderer().load(svg_bytes)
        self.svg_widget.show()

    @pyqtSlot(str)
    def display_message(self, message):
        self.message_timer.start()
        self.message_label.setText(message)

    @pyqtSlot()
    def clear_message(self):
        self.message_timer.start()
        self.message_label.setText('')

    @pyqtSlot()
    def execute_action_button(self, button):
        text = button.text()
        self.button_map[button.text()].emit()
        if text == 'Pause':
            button.setText('Unpause')
        elif text == 'Unpause':
            button.setText('Pause')


###########################################################
#               Shared methods / functions                #
###########################################################

def disconnect(control):
    '''
    Disconnects a PyQt control from its socket
    '''
    try:
        control.clicked.disconnect() 
    except TypeError:
        pass

def navigate(self, next_):
    '''
    navigate between two windows
    '''
    global prev_
    prev_ = self
    position_next_window(prev_, next_)
    self.close()
    next_.show()

def position_next_window(prev_, next_):
    '''
    Sets the position of the next_ window to the position of prev_ window
    whilst preserving the geometry of the next_ window
    '''
    w = next_.geometry().width()
    h = next_.geometry().height()
    x = round(prev_.geometry().x() + prev_.geometry().width() / 2 - w / 2)
    y = round(prev_.geometry().y() + prev_.geometry().height() / 2 - h / 2)
    next_.setGeometry(x, y, w, h)

def end_program():
    '''
    Ends the entire application
    '''
    app.quit()

def ready_check(self):
    '''
    Enables/disables a PyQt control when all vaidation checks return positive/negative
    '''
    for control in self.validated_controls:
        control.setEnabled(all(input_.valid for input_ in self.inputs.values()))

def center_window(self):
    frame_geometry = self.frameGeometry()
    screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
    centerPoint =  QApplication.desktop().screenGeometry(screen).center()
    frame_geometry.moveCenter(centerPoint)
    self.move(frame_geometry.topLeft())

def get_file(self, line_edit, message, exts_str):
    '''
    Generic dialog for selecting a file
    '''
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    fname, _ = QFileDialog.getOpenFileName(self, message, '', exts_str, options=options)
    line_edit.setText(fname)

def get_dir(self, line_edit, message='Select a directory'):
    '''
    Generic dialog for selecting a directory
    '''
    options = QFileDialog.Options() | QFileDialog.DontUseNativeDialog | QFileDialog.ShowDirsOnly
    folder = QFileDialog.getExistingDirectory(self, message, '', options=options)
    line_edit.setText(folder)

@pyqtSlot()
def validate_file(self, valid_exts):
    path = self.input_file_line_edit.text()
    if os.path.exists(os.path.normpath(path)) and path.endswith(valid_exts):
        color = VALID_COLOR
        valid = True
    else:
        color = INVALID_COLOR
        valid = False
    self.input_file_line_edit.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
    self.input_file_line_edit.valid = valid
    ready_check(self)

@pyqtSlot()
def validate_save_name(self, valid_exts):
    path = os.path.normpath(self.save_as_line_edit.text())
    dir_, fname  = os.path.split(path)
    dir_exists   = os.path.exists(dir_)
    fname_exists = os.path.exists(fname)
    if path.endswith(valid_exts) and dir_exists and not fname_exists:
        color = VALID_COLOR
        valid = True
    else:
        color = INVALID_COLOR
        valid = False
    self.save_as_line_edit.setStyleSheet(f"QLineEdit {{ background-color: {color} }}")
    self.save_as_line_edit.valid = valid
    ready_check(self)

def navigate_to_video_maker_prompt(self):
    self.pause_button.clicked.emit()
    global video_maker_prompt
    video_maker_prompt = VideoMakerPrompt(svg_dir=self.alg.output_dir)
    position_next_window(self, video_maker_prompt)
    self.alg.thread_enabled = False
    self.alg.display_signal.disconnect()
    self.alg_thread.exit()
    self.close_prompt.close()
    self.close()
    video_maker_prompt.show()


if __name__ == '__main__':

    app = QApplication([])

    ICON                = QIcon(ICON_PATH) # Needs to be run after QApplication()

    if 'Fusion' in QStyleFactory.keys():
        app.setStyle('Fusion')

    main_menu                        = MainMenu()
    gpso_setup_window                = GpsoSetupWindow()
    image_editor                     = ImageEditor()
    load_window                      = LoadWindow()
    video_maker_setup                = VideoMakerSetup()
    about_window                     = AboutWindow()

    prev_ = main_menu

    main_menu.show()

    sys.exit(app.exec_())