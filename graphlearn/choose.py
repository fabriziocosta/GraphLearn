


class Chooser:
    def choose(self, proposals,scores):
        return max(zip(proposals, scores), key=lambda x: x[1])