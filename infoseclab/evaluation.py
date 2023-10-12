from infoseclab.data import ImageNet, EPSILON, npy_uint8_to_th
from infoseclab.defenses import ResNet, ResNetDetector, RFDetector, ResNetRandom, ResNetBlur, ResNetDiscrete
from infoseclab.utils import batched_func
import torch
import numpy as np
import sklearn.metrics


class COLORS:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    RESET = '\033[0m'


def print_colored(inp, success):
    color = COLORS.GREEN if success else COLORS.RED
    print(f"{color}{inp}{COLORS.RESET}")


def accuracy(defense, images, labels=ImageNet.labels):
    """
    Compute the accuracy of a defense on a set of images.
    :param defense: The defense to evaluate.
    :param images: The images to evaluate on, of shape (N, 3, 224, 224), in the range [0, 255].
    :param labels: The labels to evaluate on.
    :return: The accuracy of the defense on the images.
    """
    with torch.no_grad():
        all_preds = batched_func(defense.classify, images, defense.device, disable_tqdm=True)
        acc = torch.mean(torch.eq(labels, all_preds).float())
    return acc


def detector_accuracy(defense_det, clean_images, adv_images):
    """
    :param defense_det: the detector
    :param clean_images: the clean ImageNet images, of shape (N, 3, 224, 224), in the range [0, 255].
    :param adv_images: the adversarial images, of shape (N, 3, 224, 224), in the range [0, 255].
    :return: the false-positive-rate (fraction of clean images detected)
    and true-positive-rate (fraction of adversarial images detected).
    """
    with torch.no_grad():
        clean_preds = batched_func(defense_det.detect, clean_images, defense_det.device, disable_tqdm=True)
        adv_preds = batched_func(defense_det.detect, adv_images, defense_det.device, disable_tqdm=True)
        fpr = torch.mean((clean_preds == 1).float())
        tpr = torch.mean((adv_preds == 1).float())

    return fpr, tpr


def assert_advs_valid(x_adv):
    """
    Assert that the adversarial images are valid.
    That is, the l_inf distance between the adversarial images and the clean
    images is less than or equal to our epsilon budget, and the images are
    in the range [0, 255].
    :param x_adv: the adversarial examples
    :return: True if the adversarial examples are valid
    """
    linf = torch.max(torch.abs(x_adv - ImageNet.clean_images))
    assert (torch.min(x_adv) >= 0.0) and (torch.max(x_adv) <= 255.0), "invalid pixel value"
    assert linf <= 1.01*EPSILON, f"linf distance too large: {linf} (target: ≤{EPSILON})"
    return True


def load_and_validate_images(path):
    """
    Load and validate the adversarial images.
    :param path: the path to the adversarial images, saved as a uint8 numpy array
    :return: True if the adversarial images are valid
    """
    x_adv = np.load(path)
    x_adv = npy_uint8_to_th(x_adv)
    assert_advs_valid(x_adv)
    return x_adv


def eval_clf(clf, x_adv, min_acc=0.99, max_adv_acc=0.02, min_target_acc=0.98, targeted=True):
    acc_clean = accuracy(clf, ImageNet.clean_images)
    success_acc = acc_clean >= min_acc
    print_colored(f"\tclean accuracy: {100*acc_clean}%", success_acc)
    assert success_acc, f"clean accuracy too low: {100*acc_clean}% (target: ≥{100*min_acc}%)"

    acc_adv = accuracy(clf, x_adv)
    success_adv = acc_adv <= max_adv_acc
    print_colored(f"\tadv accuracy: {100 * acc_adv}% (target: ≤ {100*max_adv_acc}%)", success_adv)

    success = success_acc & success_adv

    if targeted:
        acc_target = accuracy(clf, x_adv, ImageNet.targets)
        success_target = acc_target >= min_target_acc
        print_colored(f"\tadv target accuracy: {100*acc_target}% (target: ≥{100*min_target_acc}%)", success_target)
        success &= success_target
        return acc_clean, acc_adv, acc_target, success

    else:
        return acc_clean, acc_adv, success


def eval_untargeted_pgd(path="results/x_adv_untargeted.npy", device="cuda"):
    print("=== Evaluating untargeted PGD ===")
    resnet = ResNet(device)

    try:
        x_adv = load_and_validate_images(path)
    except FileNotFoundError as e:
        print("no adversarial examples found", e)
        return
    acc_clean, acc_adv, success = eval_clf(resnet, x_adv, targeted=False)

    if success:
        print_colored("SUCCESS", success)
    else:
        print_colored("NOT THERE YET!", success)


def eval_targeted_pgd(path="results/x_adv_targeted.npy", device="cuda"):
    print("=== Evaluating targeted PGD ===")
    resnet = ResNet(device)

    try:
        x_adv = load_and_validate_images(path)
    except FileNotFoundError as e:
        print("no adversarial examples found", e)
        return
    acc_clean, acc_adv, acc_target, success = eval_clf(resnet, x_adv, targeted=True)

    if success:
        print_colored("SUCCESS", success)
    else:
        print_colored("NOT THERE YET!", success)


def eval_detector_attack(path="results/x_adv_detect.npy", device="cuda"):
    print("=== Evaluating targeted PGD with Neural Network Detector ===")
    defense_det = ResNetDetector(device)

    try:
        x_adv = load_and_validate_images(path)
    except FileNotFoundError as e:
        print("no adversarial examples found", e)
        return
    acc_clean, acc_adv, acc_target, success = eval_clf(defense_det, x_adv, targeted=True)

    fpr, tpr = detector_accuracy(defense_det, ImageNet.clean_images, x_adv)

    min_fpr = 0.05
    success_fpr = fpr <= min_fpr
    assert success_fpr, f"false positive rate too high: {100*fpr}% (target: ≤{100*min_fpr}%)"
    print_colored(f"\tclean examples detected: {100 * fpr}% (target: ≤{100*min_fpr}%)", success_fpr)

    max_tpr = 0.01
    success_tpr = tpr <= max_tpr
    print_colored(f"\tadv examples detected: {100 * tpr}% (target: ≤{100*max_tpr}%)", success_tpr)

    success &= success_fpr & success_tpr

    if success:
        print_colored("SUCCESS", success)
    else:
        print_colored("NOT THERE YET!", success)


def eval_rf_detector_attack(path="results/x_adv_detect_rf.npy", device="cuda"):
    print("=== Evaluating untargeted PGD with Random Forest Detector ===")
    defense_det = RFDetector(device)

    try:
        x_adv = load_and_validate_images(path)
    except FileNotFoundError as e:
        print("no adversarial examples found", e)
        return
    acc_clean, acc_adv, success = eval_clf(defense_det, x_adv, targeted=False)

    fpr, tpr = detector_accuracy(defense_det, ImageNet.clean_images, x_adv)

    min_fpr = 0.05
    success_fpr = fpr <= min_fpr
    assert success_fpr, f"false positive rate too high: {100*fpr}% (target: ≤{100*min_fpr}%)"
    print_colored(f"\tclean examples detected: {100 * fpr}% (target: ≤{100*min_fpr}%)", success_fpr)

    max_tpr = 0.01
    success_tpr = tpr <= max_tpr
    print_colored(f"\tadv examples detected: {100 * tpr}% (target: ≤{100*max_tpr}%)", success_tpr)

    success &= success_fpr & success_tpr

    if success:
        print_colored("SUCCESS", success)
    else:
        print_colored("NOT THERE YET!", success)


def eval_blur_attack(path="results/x_adv_blur.npy", device="cuda"):
    print("=== Evaluating targeted PGD on blurred defense ===")
    defense_blur = ResNetBlur(device)

    try:
        x_adv = load_and_validate_images(path)
    except FileNotFoundError as e:
        print("no adversarial examples found", e)
        return
    acc_clean, acc_adv, acc_target, success = eval_clf(defense_blur, x_adv, min_acc=0.9, targeted=True)

    if success:
        print_colored("SUCCESS", success)
    else:
        print_colored("NOT THERE YET!", success)


def eval_random_attack(path="results/x_adv_random.npy", device="cuda"):
    print("=== Evaluating targeted PGD on randomized defense ===")
    defense_random = ResNetRandom(device)

    try:
        x_adv = load_and_validate_images(path)
    except FileNotFoundError as e:
        print("no adversarial examples found", e)
        return
    acc_clean, acc_adv, acc_target, success = eval_clf(defense_random, x_adv,
                                                       min_acc=0.9, max_adv_acc=0.02, min_target_acc=0.98, targeted=True)

    if success:
        print_colored("SUCCESS", success)
    else:
        print_colored("NOT THERE YET!", success)

def eval_discrete_attack(path="results/x_adv_discrete.npy", device="cuda"):
    print("=== Evaluating targeted PGD on discretized defense ===")
    defense_discrete = ResNetDiscrete(device)

    try:
        x_adv = load_and_validate_images(path)
    except FileNotFoundError as e:
        print("no adversarial examples found", e)
        return
    acc_clean, acc_adv, acc_target, success = eval_clf(defense_discrete, x_adv,
                                                       min_acc=0.75, max_adv_acc=0.03, min_target_acc=0.96, targeted=True)

    if success:
        print_colored("SUCCESS", success)
    else:
        print_colored("NOT THERE YET!", success)

def eval_mia(
    true_splits: np.ndarray,
    attack_scores: np.ndarray,
    ax,
    label = None,
    color = None,
    plot_decorations: bool = False
):
    if plot_decorations:
        # Plot TPR = FPR line
        ax.plot([0.0, 1.0], [0.0, 1.0], lw=0.5, c="k", ls="--")

    # Plot ROC curve
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true=true_splits, y_score=attack_scores)
    ax.plot(fpr, tpr, color=color, label=label)
    ax.loglog()
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")

    # Determine largest threshold with FPR <= 0.5% and print corresponding TPR
    target_fpr_idx = np.argmax(np.where(fpr <= 0.005, fpr, np.zeros_like(fpr)))
    print(f"{label} TPR @ FPR 0.5%: {tpr[target_fpr_idx]*100:.2f}%")

    if plot_decorations:
        # Mark target FPR
        ax.axvline(0.005, lw=0.5, c="k", ls="--")

    ax.set_ylim(top=1e0)
    ax.set_xlim(right=1e0)
