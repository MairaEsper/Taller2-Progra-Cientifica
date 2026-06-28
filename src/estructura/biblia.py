from estructura.testamento import Testamento

class Biblia:
    def __init__(self):
        self.testamentos = {
            "OT": Testamento("OT"),
            "NT": Testamento("NT")
        }
