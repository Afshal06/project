def random_flip(image):
    """Randomly flip the image horizontally."""
    if random.random() > 0.5:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def random_rotation(image):
    """Randomly rotate the image by a certain degree."""
    angle = random.randint(-30, 30)
    return image.rotate(angle)

def random_crop(image, target_size):
    """Randomly crop the image to the target size."""
    width, height = image.size
    new_width, new_height = target_size
    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)
    return image.crop((left, top, left + new_width, top + new_height))

def normalize(image):
    """Normalize the image data."""
    return (image - 127.5) / 127.5

def augment_image(image):
    """Apply a series of augmentations to the image."""
    image = random_flip(image)
    image = random_rotation(image)
    image = random_crop(image, (224, 224))  # Example target size
    image = normalize(image)
    return image