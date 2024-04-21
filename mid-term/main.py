from libs import BSM
from libs import TermStructure
from PyQt5.QtWidgets import QApplication,QWidget,QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtGui import QIcon,QImage
import sys
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


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
            price = bsm.BSPrice()
            delta = bsm.BSDelta()
            gamma = bsm.BSGamma()
            vega = bsm.BSVega()
            theta = bsm.BSTheta()
            rho = bsm.BSRho()

            result_msg = (f"1. Call Price: {price[0]:.2f}, Put Price: {price[1]:.2f}\n"
                          f"2. Call Delta: {delta[0]:.3f}, Put Delta: {delta[1]:.3f}\n"
                          f"3. Gamma: {gamma:.3f}\n"
                          f"4. Vega: {vega:.3f}\n"
                          f"5. Call Theta: {theta[0]:.3f}, Put Theta: {theta[1]:.3f}\n"
                          f"6. Call Rho: {rho[0]:.3f}, Put Rho: {rho[1]:.3f}")
            QMessageBox.information(self, 'Result', result_msg)
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Please input valid numbers.')

# Example Usage

# term_structure = TermStructure.TermStructure(base_rate=0.05)  # 5% base annual interest rate
# bsm = BSM.BlackScholesModel(s0=90, k=100, term_structure=term_structure, sigma=0.1, T=5)  # 5-year maturity

# print("Call Price:", bsm.BSPrice()[0])
# print("Put Price:", bsm.BSPrice()[1])

# print('put delta:',bsm.BSDelta()[0])
# print('call delta:',bsm.BSDelta()[1])

# print('gamma:',bsm.BSGamma())

# print("Vega:",bsm.BSVega())

# print("call Theta",bsm.BSTheta()[0])
# print("put Theta",bsm.BSTheta()[1])

# print("call Rho",bsm.BSRho()[0])
# print("put Rho",bsm.BSRho()[1])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyWidget()
    ex.show()
    sys.exit(app.exec_())