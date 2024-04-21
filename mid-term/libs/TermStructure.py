class TermStructure:
    def __init__(self, base_rate):
        self.base_rate = base_rate

    def get_rate(self, time_to_maturity):
        # This is a simple model that linearly adjusts the base rate based on the time to maturity.
        return self.base_rate + 0.01 * time_to_maturity