class RuleBasedAgent:
    """
    Simple price-threshold battery controller.
    Actions:
        0 = Charge
        1 = Discharge
        2 = Hold
    """

    def __init__(
        self,
        low_price_threshold=6.5,
        high_price_threshold=9.0,
        max_soc_percent=80.0,
        min_soc_percent=50.0
    ):
        self.low_price_threshold = low_price_threshold
        self.high_price_threshold = high_price_threshold
        self.max_soc_percent = max_soc_percent
        self.min_soc_percent = min_soc_percent

    def select_action(self, observation):
        """
        Observation:
        [normalized SOC, normalized price,
         normalized demand, normalized hour]
        """
        soc_percent = float(observation[0] * 100.0)
        price = float(observation[1] * 15.0)

        # Charge when electricity is inexpensive
        if (
            price < self.low_price_threshold
            and soc_percent < self.max_soc_percent
        ):
            return 0

        # Discharge when electricity is expensive,
        # while preserving the minimum SOC target
        if (
            price > self.high_price_threshold
            and soc_percent > self.min_soc_percent
        ):
            return 1

        return 2