import PyQt5
from PyQt5.QtWidgets import QApplication
import sys

from PyQt5.Qt import QWidget, QColor
from PyQt5.QtWidgets import QWidget
from PyQt5.Qt import QPixmap, QPainter, QPoint, QPaintEvent, QMouseEvent, QPen, \
    QColor, QSize
from PyQt5.QtCore import Qt

from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize, QCheckBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QSplitter, \
    QComboBox, QLabel, QSpinBox, QFileDialog

import try04
import os

dirname = os.path.dirname(PyQt5.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


class PaintBoard(QWidget):

    def __init__(self, Parent=None):
        '''
        Constructor
        '''
        super().__init__(Parent)

        self.__InitData()
        self.__InitView()

    def __InitData(self):

        self.__size = QSize(480, 460)

        self.__board = QPixmap(self.__size)
        self.__board.fill(Qt.white)

        self.__IsEmpty = True
        self.EraserMode = False

        self.__lastPos = QPoint(0, 0)
        self.__currentPos = QPoint(0, 0)

        self.__painter = QPainter()

        self.__thickness = 2
        self.__penColor = QColor("black")
        self.__colorList = QColor.colorNames()

    def __InitView(self):

        self.setFixedSize(self.__size)

    def Clear(self):

        self.__board.fill(Qt.white)
        self.update()
        self.__IsEmpty = True

    def ChangePenColor(self, color="black"):

        self.__penColor = QColor(color)

    def ChangePenThickness(self, thickness=10):

        self.__thickness = thickness

    def IsEmpty(self):

        return self.__IsEmpty

    def GetContentAsQImage(self):

        image = self.__board.toImage()
        return image

    def paintEvent(self, paintEvent):

        self.__painter.begin(self)

        self.__painter.drawPixmap(0, 0, self.__board)
        self.__painter.end()

    def mousePressEvent(self, mouseEvent):

        self.__currentPos = mouseEvent.pos()
        self.__lastPos = self.__currentPos

    def mouseMoveEvent(self, mouseEvent):

        self.__currentPos = mouseEvent.pos()
        self.__painter.begin(self.__board)

        if self.EraserMode == False:

            self.__painter.setPen(QPen(self.__penColor, self.__thickness))
        else:

            self.__painter.setPen(QPen(Qt.white, 10))

        # 画线
        self.__painter.drawLine(self.__lastPos, self.__currentPos)
        self.__painter.end()
        self.__lastPos = self.__currentPos

        self.update()

    def mouseReleaseEvent(self, mouseEvent):
        self.__IsEmpty = False


class MainWidget(QWidget):

    def __init__(self, Parent=None):
        '''
        Constructor
        '''
        super().__init__(Parent)

        self.__InitData()
        self.__InitView()

    def __InitData(self):

        self.__paintBoard = PaintBoard(self)

        self.__colorList = QColor.colorNames()

    def __InitView(self):

        self.setFixedSize(800, 480)
        self.setWindowTitle("Handwritten Equation Solver")

        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)

        main_layout.addWidget(self.__paintBoard)

        sub_layout = QVBoxLayout()

        sub_layout.setContentsMargins(10, 10, 10, 10)

        self.__btn_Clear = QPushButton("Clear Canvas")
        self.__btn_Clear.setParent(self)

        self.__btn_Clear.clicked.connect(self.__paintBoard.Clear)
        sub_layout.addWidget(self.__btn_Clear)

        self.__btn_Quit = QPushButton("Exit")
        self.__btn_Quit.setParent(self)
        self.__btn_Quit.clicked.connect(self.Quit)
        sub_layout.addWidget(self.__btn_Quit)

        self.__btn_Save = QPushButton("Recognize")
        self.__btn_Save.setParent(self)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)
        sub_layout.addWidget(self.__btn_Save)

        self.__cbtn_Eraser = QCheckBox("Eraser")
        self.__cbtn_Eraser.setParent(self)
        self.__cbtn_Eraser.clicked.connect(self.on_cbtn_Eraser_clicked)
        sub_layout.addWidget(self.__cbtn_Eraser)

        self.lineEdit_1 = QLabel(self)
        self.lineEdit_1.setText("Solver Recognition Equation")
        sub_layout.addWidget(self.lineEdit_1)

        self.lineEdit_2 = QLabel(self)
        self.lineEdit_2.setText("")
        sub_layout.addWidget(self.lineEdit_2)

        self.lineEdit_4 = QLabel(self)
        self.lineEdit_4.setText("Solver Recognition Equation Result")
        sub_layout.addWidget(self.lineEdit_4)

        self.lineEdit_3 = QLabel(self)
        self.lineEdit_3.setText("")
        sub_layout.addWidget(self.lineEdit_3)

        splitter = QSplitter(self)
        sub_layout.addWidget(splitter)

        self.__label_penThickness = QLabel(self)
        self.__label_penThickness.setText("Brush Size")
        self.__label_penThickness.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penThickness)

        self.__spinBox_penThickness = QSpinBox(self)
        self.__spinBox_penThickness.setMaximum(10)
        self.__spinBox_penThickness.setMinimum(2)
        self.__spinBox_penThickness.setValue(2)
        self.__spinBox_penThickness.setSingleStep(2)
        self.__spinBox_penThickness.valueChanged.connect(
            self.on_PenThicknessChange)
        sub_layout.addWidget(self.__spinBox_penThickness)

        self.__label_penColor = QLabel(self)
        self.__label_penColor.setText("Brush Color")
        self.__label_penColor.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penColor)

        self.__comboBox_penColor = QComboBox(self)
        self.__fillColorList(self.__comboBox_penColor)
        self.__comboBox_penColor.currentIndexChanged.connect(
            self.on_PenColorChange)
        sub_layout.addWidget(self.__comboBox_penColor)

        main_layout.addLayout(sub_layout)

    def __fillColorList(self, comboBox):

        index_black = 0
        index = 0
        for color in self.__colorList:
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70, 20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix), None)
            comboBox.setIconSize(QSize(70, 20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(index_black)

    def on_PenColorChange(self):
        color_index = self.__comboBox_penColor.currentIndex()
        color_str = self.__colorList[color_index]
        self.__paintBoard.ChangePenColor(color_str)

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__paintBoard.ChangePenThickness(penThickness)

    def on_btn_Save_Clicked(self):
        savePath = "./save test.jpg"
        print(savePath)
        image = self.__paintBoard.GetContentAsQImage()
        image.save(savePath)
        equation, solution = try04.calculate(savePath)
        self.lineEdit_2.setText(str(equation))
        self.lineEdit_3.setText(str(solution))

    def on_cbtn_Eraser_clicked(self):
        if self.__cbtn_Eraser.isChecked():
            self.__paintBoard.EraserMode = True
        else:
            self.__paintBoard.EraserMode = False

    def Quit(self):
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWidget = MainWidget()
    mainWidget.show()

    exit(app.exec_())
