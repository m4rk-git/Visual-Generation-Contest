import torchvision.transforms.functional as TF


def get_low_pass(tensor, kernel_size=33, sigma=2.0):
    # Returns the blurred version of the noise
    return TF.gaussian_blur(
        tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma]
    )
