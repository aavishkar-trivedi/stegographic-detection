import numpy as np

def adversarial_check(image):
    noise = np.std(image)

    if noise > 0.25:
        return {
            "status": "ADVERSARIAL",
            "score": 0.8
        }

    elif noise > 0.15:
        return {
            "status": "SUSPICIOUS",
            "score": 0.5
        }

    else:
        return {
            "status": "CLEAN",
            "score": 0.1
        }
