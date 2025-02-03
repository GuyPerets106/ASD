import numpy as np
from PIL import Image


class BadNets(object):
    """The BadNets [paper]_ backdoor transformation. Inject a trigger into an image (ndarray with
    shape H*W*C) to get a poisoned image (ndarray with shape H*W*C).

    Args:
        trigger_path (str): The path of trigger image whose background is in black.

    .. rubric:: Reference

    .. [paper] "Badnets: Evaluating backdooring attacks on deep neural networks."
     Tianyu Gu, et al. IEEE Access 2019.
    """

    def __init__(self, trigger_path):
        with open(trigger_path, "rb") as f:
            trigger_ptn = Image.open(f).convert("RGB")
        self.trigger_ptn = np.array(trigger_ptn)
        self.trigger_loc = np.nonzero(self.trigger_ptn)

    def __call__(self, img):
        return self.add_trigger(img)

    def add_trigger(self, img):
        if not isinstance(img, np.ndarray):
            raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
        if len(img.shape) != 3:
            raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
        img[self.trigger_loc] = 0
        poison_img = img + self.trigger_ptn

        return poison_img

class SIG(object):
    """
    The SIG (Sinusoidal Signal) backdoor transformation. Adds a sinusoidal trigger
    to an image (ndarray with shape H*W*C) to get a poisoned image (ndarray with shape H*W*C).

    Args:
        amplitude (float): The amplitude of the sinusoidal perturbation.
        frequency (float): The frequency of the sinusoidal perturbation.

    Reference:
        "A new Backdoor Attack in CNNs by training set corruption without label poisoning"
        Paper link: https://arxiv.org/pdf/1902.11237
    """

    def __init__(self, amplitude=20, frequency=6):
        self.amplitude = amplitude
        self.frequency = frequency

    def __call__(self, img):
        return self.add_trigger(img)

    def add_trigger(self, img):
        """
        Add the sinusoidal trigger to the image.

        Args:
            img (np.ndarray): Input image with shape (H, W, C) and pixel values in [0, 255].

        Returns:
            poison_img (np.ndarray): Poisoned image with shape (H, W, C) and pixel values in [0, 255].
        """
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Img should be np.ndarray. Got {type(img)}")
        if len(img.shape) != 3:
            raise ValueError(f"The shape of img should be HWC. Got {img.shape}")
        if img.max() > 255 or img.min() < 0:
            raise ValueError("Img pixel values should be in the range [0, 255].")

        h, w, _ = img.shape
        x = np.linspace(0, 2 * np.pi, w)
        y = np.linspace(0, 2 * np.pi, h)
        X, Y = np.meshgrid(x, y)
        sine_wave = self.amplitude * np.sin(self.frequency * X + self.frequency * Y)
        sine_wave = sine_wave.astype(np.float32)

        # Add the sinusoidal wave to all channels and clip to valid range [0, 255]
        poison_img = img + sine_wave[:, :, np.newaxis]
        poison_img = np.clip(poison_img, 0, 255)

        return poison_img.astype(np.uint8)
    
class InvisibleGrid(object):
    """
    Overlays a subtle sinusoidal grid pattern on an image (H*W*C).
    The amplitude/frequency can be tuned to make the grid faint yet learnable.
    """

    def __init__(self, amplitude=8.0, frequency=8.0):
        """
        Args:
            amplitude (float): Peak value to add/subtract for the grid.
            frequency (float): Controls the 'period' of the grid.
        """
        self.amplitude = amplitude
        self.frequency = frequency

    def __call__(self, img):
        return self.add_trigger(img)

    def add_trigger(self, img):
        """
        Args:
            img (np.ndarray): Shape (H, W, C), pixel values in [0, 255].
        Returns:
            poison_img (np.ndarray): Same shape, with the grid pattern added.
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
        if len(img.shape) != 3:
            raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))

        H, W, C = img.shape

        # Create meshgrid for the sine-wave-based grid
        y = np.arange(H).reshape(-1, 1)
        x = np.arange(W).reshape(1, -1)
        # A simple 2D sinusoidal pattern
        grid_pattern = (
            self.amplitude
            * (np.sin(2 * np.pi * y / self.frequency) + np.sin(2 * np.pi * x / self.frequency))
            / 2
        )
        # Expand to match channels
        grid_pattern = np.repeat(grid_pattern[:, :, np.newaxis], C, axis=2)

        poison_img = img.astype(np.float32) + grid_pattern
        poison_img = np.clip(poison_img, 0, 255)

        return poison_img.astype(np.uint8)
    
class SmoothPoison(object):
    """
    Adds a smooth (low-frequency) perturbation to each poisoned image.
    This is often called "Smooth Poison" because it seamlessly modifies
    the entire image in a subtle way.

    Args:
        amplitude (float): Scale of the perturbation.
        kernel_size (int): The size of the smoothing kernel. Larger = smoother.
    """

    def __init__(self, amplitude=15.0, kernel_size=15):
        self.amplitude = amplitude
        self.kernel_size = kernel_size

    def __call__(self, img):
        return self.add_trigger(img)

    def add_trigger(self, img):
        """
        Args:
            img (np.ndarray): Shape (H, W, C) with pixel values in [0, 255].
        Returns:
            poison_img (np.ndarray): The 'smoothly' perturbed image.
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
        if len(img.shape) != 3:
            raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))

        H, W, C = img.shape
        # 1) Create random noise
        noise = np.random.rand(H, W).astype(np.float32) * 2 - 1  # range [-1, 1]

        # 2) Smooth the noise via a simple convolution or Gaussian filter
        # Example below: naive manual blur. For something more advanced, use scipy's gaussian_filter.
        for _ in range(3):
            # One or more passes of a simple blur to lower the frequency
            noise = 0.25 * (np.roll(noise, 1, axis=0) + np.roll(noise, -1, axis=0)
                            + np.roll(noise, 1, axis=1) + np.roll(noise, -1, axis=1))

        # 3) Scale and replicate noise across channels
        smooth_perturb = (noise * self.amplitude)
        smooth_perturb = np.repeat(smooth_perturb[:, :, np.newaxis], C, axis=2)

        # 4) Add to the image, clip to valid [0, 255]
        poison_img = img.astype(np.float32) + smooth_perturb
        poison_img = np.clip(poison_img, 0, 255)

        return poison_img.astype(np.uint8)