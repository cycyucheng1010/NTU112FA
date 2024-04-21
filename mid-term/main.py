from libs import BSM
from libs import TermStructure
from PyQt5.QtWidgets import QApplication,QWidget,QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtGui import QIcon,QImage
import sys

class MyWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Black-Scholes Option Price Calculator')
        self.setGeometry(300, 300, 500, 500)
        self.setWindowIcon(QIcon('logo.ico'))
        self.layout = QVBoxLayout()
        self.setInputField()
        self.setButton()
        self.setLayout(self.layout)

    def setButton(self):
        self.btn = QPushButton('Calculate', self)
        self.btn.clicked.connect(self.calculate)
        self.layout.addWidget(self.btn)
        #self.btn.show()

    def createLineEdit(self, label_text):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        line_edit = QLineEdit()
        layout.addWidget(label)
        layout.addWidget(line_edit)
        return layout, line_edit

    def setInputField(self):
        self.s0_layout, self.s0_edit = self.createLineEdit("S0 (Current Stock Price):")
        self.k_layout, self.k_edit = self.createLineEdit("K (Strike Price):")
        self.r_layout, self.r_edit = self.createLineEdit("Base Rate (Annual Risk-free Rate):")
        self.sigma_layout, self.sigma_edit = self.createLineEdit("Sigma (Volatility):")
        self.T_layout, self.T_edit = self.createLineEdit("T (Time to Maturity in years):")
        self.s0_edit.setText('90')
        self.k_edit.setText('100')
        self.r_edit.setText('0.05')
        self.sigma_edit.setText('0.1')
        self.T_edit.setText('5')
        
        self.layout.addLayout(self.s0_layout)
        self.layout.addLayout(self.k_layout)
        self.layout.addLayout(self.r_layout)
        self.layout.addLayout(self.sigma_layout)
        self.layout.addLayout(self.T_layout)
    
    def calculate(self):
        try:
            s0 = float(self.s0_edit.text())
            k = float(self.k_edit.text())
            r = float(self.r_edit.text())
            sigma = float(self.sigma_edit.text())
            T = float(self.T_edit.text())

            term_structure = TermStructure.TermStructure(r)
            bsm = BSM.BlackScholesModel(s0, k, term_structure, sigma, T)
            self.price = bsm.BSPrice()
            self.delta = bsm.BSDelta()
            self.gamma = bsm.BSGamma()
            self.vega = bsm.BSVega()
            self.theta = bsm.BSTheta()
            self.rho = bsm.BSRho()

            result_msg = (f"1. Call Price: {self.price[0]:.2f}, Put Price: {self.price[1]:.2f}\n"
                          f"2. Call Delta: {self.delta[0]:.3f}, Put Delta: {self.delta[1]:.3f}\n"
                          f"3. Gamma: {self.gamma:.3f}\n"
                          f"4. Vega: {self.vega:.3f}\n"
                          f"5. Call Theta: {self.theta[0]:.3f}, Put Theta: {self.theta[1]:.3f}\n"
                          f"6. Call Rho: {self.rho[0]:.3f}, Put Rho: {self.rho[1]:.3f}")
            QMessageBox.information(self, 'Result', result_msg)
            self.terminalShow()
        except ValueError:
            print('Error','Please input valid numbers.')
            QMessageBox.warning(self, 'Error', 'Please input valid numbers.')

    def terminalShow(self):
        print("Call Price:", self.price[0])
        print("Put Price:", self.price[0])

        print('put delta:',self.delta[0])
        print('call delta:',self.delta[1])

        print('gamma:',self.gamma)

        print("Vega:",self.vega)

        print("call Theta",self.theta[0])
        print("put Theta",self.theta[1])

        print("call Rho",self.rho[0])
        print("put Rho",self.rho[1])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyWidget()
    ex.show()
    sys.exit(app.exec_())