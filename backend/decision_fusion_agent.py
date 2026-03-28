from .config import WEIGHTS

def fuse_decision(feature_score, dl_score, adv_score):
    final_score = (
        WEIGHTS["feature"] * feature_score +
        WEIGHTS["deep_learning"] * dl_score +
        WEIGHTS["adversarial"] * adv_score
    )

    if final_score > 0.5:
        verdict = "STEGO IMAGE DETECTED"
    else:
        verdict = "CLEAN IMAGE"

    return final_score, verdict
