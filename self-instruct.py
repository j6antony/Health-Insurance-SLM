from typing import Literal

#Mode options
#1. Random: randomly select 8 out of 18 categories with an example from each
#2. Target: Pick examples from specified categories
#3. Track which categories have been generated for and pick the underrepresented category examples

class selfinstruct:
    def __init__(self, seed_path, mode: Literal["random", "target", "balanced"]="balanced"):
        self.seed_path = seed_path
