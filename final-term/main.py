import sys
import time
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QLabel
from PyQt5.QtGui import QIcon
import QuantLib as ql

class OptionCalculator(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Option Risk Capital Calculator')
        self.setGeometry(300, 300, 500, 500)
        self.setWindowIcon(QIcon('logo.ico'))
        layout = QVBoxLayout()
        formLayout = QFormLayout()

        # Creating input fields for various parameters with default values
        self.s_input = QLineEdit('100')
        self.k_input = QLineEdit('100')
        self.t_input = QLineEdit('9')
        self.r1m_input = QLineEdit('0.02')
        self.r3m_input = QLineEdit('0.0225')
        self.r6m_input = QLineEdit('0.025')
        self.r12m_input = QLineEdit('0.028')
        self.sigma6m_input = QLineEdit('0.25')
        self.sigma1y_input = QLineEdit('0.30')
        self.stock_position_input = QLineEdit('1000')
        self.option_position_input = QLineEdit('1000')
        
        # Adding input fields to the form layout
        formLayout.addRow('Stock Price (S):', self.s_input)
        formLayout.addRow('Strike Price (K):', self.k_input)
        formLayout.addRow('Time to Maturity (T in months):', self.t_input)
        formLayout.addRow('1M Risk-Free Rate:', self.r1m_input)
        formLayout.addRow('3M Risk-Free Rate:', self.r3m_input)
        formLayout.addRow('6M Risk-Free Rate:', self.r6m_input)
        formLayout.addRow('12M Risk-Free Rate:', self.r12m_input)
        formLayout.addRow('6M Implied Volatility:', self.sigma6m_input)
        formLayout.addRow('1Y Implied Volatility:', self.sigma1y_input)
        formLayout.addRow('Stock Position:', self.stock_position_input)
        formLayout.addRow('Option Position:', self.option_position_input)
        
        self.calculate_button = QPushButton('Calculate')
        self.result_label = QLabel('Result will be shown here.')

        # Connecting button click to the calculation function
        self.calculate_button.clicked.connect(self.calculate_risk_capital)

        layout.addLayout(formLayout)
        layout.addWidget(self.calculate_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def calculate_risk_capital(self):
        start_time = time.time()

        # Reading input values
        S = float(self.s_input.text())
        K = float(self.k_input.text())
        T = float(self.t_input.text()) / 12.0
        r1M = float(self.r1m_input.text())
        r3M = float(self.r3m_input.text())
        r6M = float(self.r6m_input.text())
        r12M = float(self.r12m_input.text())
        sigma_6M = float(self.sigma6m_input.text())
        sigma_1Y = float(self.sigma1y_input.text())
        stock_position = float(self.stock_position_input.text())
        option_position = float(self.option_position_input.text())

        # Interpolating risk-free rate for 9 months
        r = r3M + (r6M - r3M) * (T - 3.0/12.0) / (6.0/12.0 - 3.0/12.0)
        # Interpolating implied volatility for 9 months
        sigma = sigma_6M + (sigma_1Y - sigma_6M) * (T - 6.0/12.0) / (12.0/12.0 - 6.0/12.0)

        # Setting the calculation date
        calculation_date = ql.Date().todaysDate()
        ql.Settings.instance().evaluationDate = calculation_date

        # Creating QuantLib objects for the calculation
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, r, ql.Actual365Fixed()))
        volatility = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date, ql.NullCalendar(), sigma, ql.Actual365Fixed()))

        # Defining the payoff and exercise of the option
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
        exercise = ql.EuropeanExercise(calculation_date + ql.Period(int(T*365), ql.Days))

        # Creating the European option
        european_option = ql.EuropeanOption(payoff, exercise)
        bsm_process = ql.BlackScholesMertonProcess(spot_handle, flat_ts, flat_ts, volatility)

        # Setting the pricing engine
        european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))

        # Calculating option price and risk measures
        option_price = european_option.NPV()
        delta = european_option.delta()
        vega = european_option.vega()

        # Calculating market risk capital
        market_risk_capital = (delta * stock_position * S) + (vega * option_position * sigma)

        end_time = time.time()
        execution_time = end_time - start_time

        # Displaying the results
        self.result_label.setText(f"Option Price: {option_price}\nDelta: {delta}\nVega: {vega}\nMarket Risk Capital: {market_risk_capital}\nExecution Time: {execution_time:.8f} seconds")

def main():
    app = QApplication(sys.argv)
    ex = OptionCalculator()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
