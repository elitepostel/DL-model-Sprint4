import torchvision.transforms as T


# ==============================

#   НОРМАЛИЗАЦИЯ ДЛЯ RESNET50

# ==============================

# Стандартные mean/std под ImageNet

IMAGENET_MEAN = [0.485, 0.456, 0.406]

IMAGENET_STD = [0.229, 0.224, 0.225]





# ==============================

#    TRAIN TRANSFORMS

# ==============================

def get_train_transforms(img_size: int = 224):

    """

    Преобразования для обучения.

    - Аугментации: случайные повороты, кропы, флипы

    - Масштабирование до одного размера

    - Нормализация под ResNet

    """

    return T.Compose([

        # Изменение размера изображения

        T.Resize((img_size, img_size)),



        # Аугментации — важны для уменьшения переобучения

        T.RandomHorizontalFlip(p=0.5),

        T.RandomVerticalFlip(p=0.1),

        T.RandomRotation(degrees=15),



        # Конвертация в Tensor и нормализация

        T.ToTensor(),

        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

    ])





# ==============================

#    VALIDATION TRANSFORMS

# ==============================

def get_val_transforms(img_size: int = 224):

    """

    Преобразования для валидации.

    Никаких аугментаций, только:

    - Resize

    - ToTensor

    - Normalize

    """

    return T.Compose([

        T.Resize((img_size, img_size)),

        T.ToTensor(),

        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

    ])





# ==============================

#       TEST TRANSFORMS

# ==============================

def get_test_transforms(img_size: int = 224):

    """

    Преобразования для финального инференса.

    Полное совпадение с валидацией.

    """

    return T.Compose([

        T.Resize((img_size, img_size)),

        T.ToTensor(),

        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

    ])