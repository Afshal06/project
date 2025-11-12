class LeafDataset:
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.transform = transform
        self.images, self.labels = self.load_data()

    def load_data(self):
        # Load images and labels from the annotation file
        images = []
        labels = []
        with open(self.annotation_file, 'r') as file:
            for line in file:
                image_path, label = line.strip().split(',')
                images.append(image_path)
                labels.append(label)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = self.load_image(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def load_image(self, image_path):
        # Load and preprocess the image
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        return image