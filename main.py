import sys
from matplotlib.figure import Figure
from matplotlib import patches
import SimpleITK as sitk
import os
import csv
from Fourie import FourieMaster
from scipy.spatial import ConvexHull
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, uic, QtCore,QtGui
from PyQt5.QtWidgets import QFileDialog,QMessageBox,QSlider
import matplotlib as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from Contours import ContorProduce
UI_PATH = 'view.ui'
if not os.path.exists('data'):
    os.mkdir('data')
SAVE_PATH = 'data/sliced.npy'
StyleSheet = '''
        *{
            font-weight: bold;
        }
        QMainWindow{
            background-color:rgba(148,87,168,0.5);
        }
        QLabel{
            background-color:rgba(0,0,0,0.5);
            color: white;
        }
        QLineEdit{
            border-radius:8px;
            background-color:rgba(0,0,0,0.5);
            color: white;
        }
        QLineEdit:hover{
        }
        QPushButton{
            background-color: rgba(0,0,0,0.5);
            border-color: white;
            color :white;
        }
        QPushButton:hover{
            padding-top: 10px;
            padding-bottom: 5px;
        }
        QComboBox{
            background-color: rgba(0,0,0,0.1);
            color :black;
        }
        QSlider{
            background-color: rgba(0,0,0,0.5);

        }
                
        '''

def loadNII(file_path): #return numpy array
    tmp = os.path.splitext(file_path)
    if not(tmp[1] == '.nii' or tmp[1]== '.gz'):
       return np.array([]) 
    image = sitk.ReadImage(file_path)
    ndrev = sitk.GetArrayFromImage(image)
    return ndrev

class Application(QtWidgets.QMainWindow):
    def __init__(self):
        # Call the inherited classes __init__ method
        super().__init__()
        #UIの初期化

        self.initUI()
        self.Setting()
        self.initFigure()
        self.initContorFigure()
        
    def Setting(self):
        self.anno = False
        self.setStyleSheet(StyleSheet)
        self.Loaded = False
        self.kParameterWidget.setValidator(QtGui.QIntValidator())
        self.FileBrowser.clicked.connect(self.showDIALOG)
        self.NiiList.currentIndexChanged.connect(lambda: self.showNii(self.NiiList.currentText()))
        self.ContorList.currentIndexChanged.connect(lambda: self.showContor(self.ContorList.currentText()))
        self.Button.clicked.connect(self.AnalizeFourie)
        self.saveButton.clicked.connect(self.SaveAsCsv)
        
    def initUI(self):
        self.resize(1400,800)#ウィンドウサイズの変更
        self.FigureWidget = QtWidgets.QWidget(self)
        self.FigureWidget.setGeometry(10,50,600,600) 
        # FigureWidgetにLayoutを追加
        self.FigureLayout = QtWidgets.QVBoxLayout(self.FigureWidget)
        self.FigureLayout.setContentsMargins(0,0,0,0)
        #Contorを表示するwidgetを追加
        self.ContorFigureWidget = QtWidgets.QWidget(self)
        self.ContorFigureWidget.setGeometry(750,50,600,600)
       #ContorWidget用のlayoutを追加 
        self.ContorFigureLayout = QtWidgets.QVBoxLayout(self.ContorFigureWidget)
        self.ContorFigureLayout.setContentsMargins(0,0,0,0)


        self.ContorList = QtWidgets.QComboBox(self)
        self.ContorList.setGeometry(750,10,50,20)

        self.FileBrowser = QtWidgets.QPushButton('select file',self)
        self.FileBrowser.move(0,10)

        self.NiiList =QtWidgets.QComboBox(self) 
        self.NiiList.setGeometry(550,10,50,20)

        self.FileNameText = QtWidgets.QLabel(self)
        self.FileNameText.setText('your selected file path')
        self.FileNameText.setGeometry(110,10,300,30)

        self.Button = QtWidgets.QPushButton('Fourie',self)
        self.Button.setGeometry(820,10,100,30)
        
        self.saveButton = QtWidgets.QPushButton('save',self)
        self.saveButton.setGeometry(1000,10,100,30)

        self.kParameterWidget = QtWidgets.QLineEdit(self)
        self.kParameterWidget.setGeometry(950,10,30,30)

    def initSlider(self,vmax):
        self.sld = QtWidgets.QSlider(Qt.Vertical,self)
        self.sld.setMinimum(0)
        self.sld.setMaximum(vmax)
        self.sld.setFocusPolicy(Qt.NoFocus)
        self.sld.setGeometry(650,50,20,600)
        self.sld.setValue(0)
        self.sld.setSingleStep(1)
        self.sld.valueChanged.connect(self.valueChange)
        self.sld.show()

    def valueChange(self):
        self.showNii(str(self.sld.value()))


    def initContorFigure(self):
        self.ContorFigure = plt.figure.Figure()
        self.ContorFigureCanvas = FigureCanvas(self.ContorFigure)
        self.ContorFigureLayout.addWidget(self.ContorFigureCanvas)
        self.contor_axes = self.ContorFigure.add_subplot(1,1,1)
        self.contor_axes.set_aspect('equal')
        self.contor_axes.axis('off')
    
    def initFigure(self):
        self.Figure = plt.figure.Figure()
        # FigureをFigureCanvasに追加
        self.FigureCanvas = FigureCanvas(self.Figure)
        # LayoutにFigureCanvasを追加
        self.FigureLayout.addWidget(self.FigureCanvas)
        #figureからaxesを作成
        self.axes = self.Figure.add_subplot(1,1,1)
        self.axes.axis('off')
    
    def updateFigure(self):
        self.FigureCanvas.draw()

    def updateContorFigure(self):
        self.ContorFigureCanvas.draw()
    

    def showDIALOG(self):
        self.NiiList.clear()
        self.NiiLength = 0
        # 第二引数はダイアログのタイトル、第三引数は表示するパス
        fname = QFileDialog.getOpenFileName(self, 'Open file', './')
        FILEPATH = fname[0]
        self.FileNameText.setText(FILEPATH)
        NII_Data = loadNII(FILEPATH)
        if len(NII_Data) > 0:
            self.NiiLength = len(NII_Data) 
            np.save(SAVE_PATH,NII_Data)
            for index in range(self.NiiLength):
                self.NiiList.addItem(str(index))
            self.initSlider(self.NiiLength-1)
        else:
            dlg = QMessageBox(self)
            dlg.setWindowTitle('error')
            dlg.setText('input file  needs to be .nii file')
            dlg.exec()

    def showNii(self,index):#indexがstr型でくる
        self.axes.cla()
        self.axes.axis('off')
        self.ContorList.clear()
        if index == '':
            return
        index = int(index)
        self.NII_IMAGE= np.load(SAVE_PATH)[index]
        self.ContorData = ContorProduce(self.NII_IMAGE)#画像選択時にその輪郭データを作成
        for i in range(len(self.ContorData.contours)):
            self.ContorList.addItem(str(i))
        tmp = self.NII_IMAGE
        self.axes.imshow(self.NII_IMAGE)
        self.updateFigure()
        
    def showContor(self,index):#indexがstr型でくる
        self.Loaded = True
        self.contor_axes.cla()#前のplotデータの削除
        if index == '':
            return
        index = int(index)
        self.ContorBox = self.ContorData.produce(index)
        X = self.ContorBox[:,0]
        Y = self.ContorBox[:,1]
        self.anno = self.contor_axes.scatter(X,Y,c='blue',s=10)
        self.contor_axes.axis('off')
        self.updateContorFigure()

    def AnalizeFourie(self):
        #length = self.ContorData.culcArclength()
        if self.Loaded == False:
            return
        self.fourie = FourieMaster(self.ContorBox)
        self.fourie.constMatrix()
        self.fourieMatrix = self.fourie.matrix
        K = int(self.kParameterWidget.text())
        if K >= len(self.ContorBox):
            return
        self.FouriePoint = self.fourie.reconstract(K)
        self.fouriePlot()

    def fouriePlot(self):
        self.contor_axes.cla()
        X = self.FouriePoint.real
        Y = self.FouriePoint.imag
        self.contor_axes.scatter(X,Y,c='blue',s=10)
        self.contor_axes.axis('off')
        self.updateContorFigure()

    def SaveAsCsv(self):
        file_name, filters = QFileDialog.getSaveFileName(filter="CSV files (*.csv)")
        # 保存ボタン押した後の処理
        if file_name != '':
            np.savetxt(file_name,self.fourieMatrix)
        return


        
def main():
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = Application()
    mainwindow.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()