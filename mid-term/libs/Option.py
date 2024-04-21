class Option:
    def __init__(self, strike, maturity, option_type='call'):
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type