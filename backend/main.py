import argparse

from .input_handler import load_image
from .feature_agent import extract_features
from .deep_learning_agent import deep_learning_detection
from .adversarial_agent import adversarial_check
from .decision_fusion_agent import fuse_decision


def parse_args():
    parser = argparse.ArgumentParser(description="Agentic steganalysis detector")
    parser.add_argument(
        "image_path",
        nargs="?",
        default="image.jpg",
        help="Path to input image (default: image.jpg)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image = load_image(args.image_path)

    features = extract_features(image)
    feature_score = features["feature_score"]

    dl_score = deep_learning_detection(image)

    adv = adversarial_check(image)
    adv_score = adv["score"]

    final_score, verdict = fuse_decision(
        feature_score,
        dl_score,
        adv_score,
    )

    print("Feature Score:", feature_score)
    print("Deep Learning Score:", dl_score)
    print("Adversarial Status:", adv["status"])
    print("Final Score:", final_score)
    print("Verdict:", verdict)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(exc)
        print("Usage: python main.py <path_to_image>")
