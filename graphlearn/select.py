


class SelectMax:
    def select(self, proposals,scores):
        return max(zip(proposals, scores), key=lambda x: x[1])
