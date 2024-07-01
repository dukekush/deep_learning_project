import os
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.optim import SGD
from misc_functions import recreate_image, save_image
from torchvision.transforms.functional import rotate
from torchvision import transforms
import cv2
from PIL import Image, ImageDraw, ImageFilter
import tqdm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from sklearn.metrics import confusion_matrix
import itertools
import copy
from re import findall
from torchcam.methods import ScoreCAM
import io
import base64
from IPython.display import display, HTML


use_gpu = torch.backends.mps.is_available()


def train(
        model, 
        optimizer,
        loss_function,
        train_loader,
        validation_loader,
        num_epochs,
        lr_scheduler=None,
        patience=15,
        device=torch.device("cpu")
    ):
    '''
    Returns the best model from training, and the loss and accuracy for each epoch.

            Args:
                    model (torch.nn.Module): The model to train
                    optimizer (torch.optim): The optimizer to use
                    loss_function (torch.nn): The loss function to use
                    train_loader (torch.utils.data.DataLoader): The training data
                    validation_loader (torch.utils.data.DataLoader): The validation data
                    num_epochs (int): The number of epochs to train for
                    lr_scheduler (torch.optim.lr_scheduler): The learning rate scheduler to use
                    patience (int): The number of epochs to wait for validation loss to improve before stopping
                    device (torch.device): The device to use for training

            Returns:
                    best_model (torch.nn.Module): The best model from training
                    train_loss_list_per_epoch (list): The training loss for each epoch
                    train_accuracy_per_epoch (list): The training accuracy for each epoch
                    val_loss_list_per_epoch (list): The validation loss for each epoch
                    val_accuracy_per_epoch (list): The validation accuracy for each epoch
    '''
    model.to(device)
    min_val_loss = float('inf')
    val_acc = 0
    val_loss = torch.inf
    # Create lists to store losses and accuracy
    train_loss_list_per_epoch = []
    train_accuracy_per_epoch = []
    val_loss_list_per_epoch = []
    val_accuracy_per_epoch = []
    for epoch in range(num_epochs):
        
        model.train()

        # Create lists to store losses and accuracy
        train_loss_list_per_itr = []
        train_accuracy_per_itr = []

        loop = tqdm.tqdm(train_loader)
        for i, (images, labels) in enumerate(loop):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            correct = predictions.eq(labels).sum().item()
            accuracy = correct / len(labels)

            # if i % 10 ==0:
            train_loss_list_per_itr.append(loss.item())
            train_accuracy_per_itr.append(accuracy)

            # Print statistics
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(train_loss=np.mean(train_loss_list_per_itr), val_loss=val_loss, train_acc=100.*np.mean(train_accuracy_per_itr), val_acc=100.*val_acc)
        
        # Save loss and accuracy for each epoch
        train_loss_list_per_epoch.append(np.mean(train_loss_list_per_itr))
        train_accuracy_per_epoch.append(np.mean(train_accuracy_per_itr))

        # Evaluate model for each epoch
        val_loss, val_acc = evaluation(model, validation_loader, loss_function, device)
        val_loss_list_per_epoch.append(val_loss)
        val_accuracy_per_epoch.append(val_acc)

        # Update learning rate
        if lr_scheduler:
            lr_scheduler.step(val_loss)

        # Early stopping
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            no_improvement = 0
            best_model = copy.deepcopy(model)
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print('Early stopping!')
                break

    return best_model, train_loss_list_per_epoch, train_accuracy_per_epoch, val_loss_list_per_epoch, val_accuracy_per_epoch


def evaluation(model, data_loader, loss_critetion, device):
    '''
    Returns the loss and accuracy for the given model and data loader.

            Args:
                    model (torch.nn.Module): The model to evaluate
                    data_loader (torch.utils.data.DataLoader): The data to evaluate on
                    loss_critetion (torch.nn): The loss function to use
                    device (torch.device): The device to use for evaluation

            Returns:
                    val_loss (float): The validation loss
                    accuracy (float): The validation accuracy

    '''
    model.eval()
    correct = 0
    total = 0
    val_loss = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_critetion(outputs, labels)
            val_loss.append(loss.item())

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return np.mean(val_loss), accuracy


def plot_accuracy(train_accuracies, val_accuracies, title, filename):
    '''
    Shows a plot of the training and validation accuracy.

            Args:
                    train_accuracies (list): The training accuracy for each epoch
                    val_accuracies (list): The validation accuracy for each epoch
                    title (str): The title of the plot
                    filename (str): The filename to save the plot to
            
            Returns:
                    None
    '''
    epochs = range(1, len(train_accuracies) + 1)
    
    plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(filename)
    plt.show()


def plot_loss(train_losses, val_losses, title, filename):
    '''
    Shows a plot of the training and validation loss.

            Args:
                    train_losses (list): The training loss for each epoch
                    val_losses (list): The validation loss for each epoch
                    title (str): The title of the plot
                    filename (str): The filename to save the plot to
            
            Returns:
                    None
    '''
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(filename)
    plt.show()


def generate_confusion_matrix(model, data_loader, device):
    '''
    Returns the confusion matrix for the given model and data loader (dataset).

            Args:
                    model (torch.nn.Module): The model to evaluate
                    data_loader (torch.utils.data.DataLoader): The data to evaluate on
                    device (torch.device): The device to use for evaluation

            Returns:
                    cm (numpy.ndarray): The confusion matrix
    '''
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return confusion_matrix(all_labels, all_preds)


def plot_confusion_matrix(cm, classes, title, filename, normalize=False, cmap=plt.cm.Blues):
    '''
    Shows a plot of the confusion matrix.

            Args:
                    cm (numpy.ndarray): The confusion matrix
                    classes (list): The classes to use
                    title (str): The title of the plot
                    filename (str): The filename to save the plot to
                    normalize (bool): Whether to normalize the confusion matrix
                    cmap (matplotlib.colors.Colormap): The colormap to use

            Returns:
                    None
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    plt.show()
    

def add_box(image, position, size, color):
    '''
    Adds a square box to the given image.

            Args:
                    image (PIL.Image): The image to add the box to
                    position (tuple): The position of the box
                    size (int): The size of the box
                    color (tuple): The color of the box, from 0 to 1
            
            Returns:
                    image (PIL.Image): The image with the box added

    '''
    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Draw the box
    draw.rectangle([position, (position[0] + size, position[1] + size)], fill=color*255)

    return image


class AddBoxTransform:
    '''
    A class to add a square box to an image. Works with torchvision.transforms.Compose.


    Attributes
    ----------
    position : tuple
        The position of the box
    size : int
        The size of the box
    color : tuple
        The color of the box, from 0 to 1
    '''

    def __init__(self, position, size, color):
        self.position = position
        self.size = size
        self.color = color

    def __call__(self, image):
        return add_box(image, self.position, self.size, self.color)
    

class SemiSupervisedDataset(Dataset):
    '''
    A class to represent a semi-supervised dataset. The dataset is created by adding a box to each image in the given dataset or by rotating each image in the given dataset.
    Works with torch.utils.data.DataLoader.

    Attributes
    ----------
    root : str
        The root directory of the dataset
    classes : list
        The classes of the dataset
    class_to_idx : dict
        A dictionary mapping each class to its index
    image_transformation : str
        The image transformation to apply to the dataset (either 'black-white' or 'rotate').
        'black-white' means that a black or white box is added to each image.
        'rotate' means each image is rotated by 0 degrees, 90 degrees, 180 degrees or 270 degrees.
    box_size : int
        The size of the box to add to each image (default is 10)
    transform : torchvision.transforms.Compose
        The transformations to apply to each image (default is None)
    image_paths : list
        A list of the paths of each image in the dataset
    class_labels : list
        A list of the labels of each image in the dataset
    '''

    def __init__(self, root, image_transformation, transform=None, box_size=10):
        self.root = root
        self.classes = ['black', 'white'] if image_transformation == 'black-white' else ['0_degrees', '90_degrees', '180_degrees', '270_degrees']
        self.class_to_idx = {'black':0, 'white':1} if image_transformation == 'black-white' else {c : int(findall(r'\d+', c)[0]) // 90 for c in self.classes}
        self.image_transformation = image_transformation
        self.box_size = box_size
        self.transform = transform
        self.image_paths = []
        self.class_labels = []

        for class_label, class_name in enumerate(sorted(os.listdir(root))):
            class_dir = os.path.join(root, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        if image_transformation == 'black-white':
                            for color in self.classes:
                                for _ in range(1):  # repeat few times to get more training data
                                    self.image_paths.append(os.path.join(class_dir, file_name))
                                    self.class_labels.append(color)
                        elif image_transformation == 'rotation':
                            for angle in self.classes:
                                self.image_paths.append(os.path.join(class_dir, file_name))
                                self.class_labels.append(angle)
                        else:
                            raise ValueError('image transformation must be one of "black-white" or "rotation"')


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, index):
        '''
        Returns the image and its label at the given index.

                Args:
                        index (int): The index of the image to return

                Returns:
                        img (PIL.Image) or (torch.Tensor): The image at the given index, return depends on the value of self.transform.
                        img_class (int): The index of the class of the image at the given index
        '''
        img_path = self.image_paths[index]
        img_class = self.class_labels[index]

        img = Image.open(img_path)
        
        if self.image_transformation == 'black-white':
            position = (np.random.randint(0, img.size[0]-self.box_size), np.random.randint(0, img.size[1]-self.box_size))
            perturbation = AddBoxTransform(position, self.box_size, self.class_to_idx[img_class])
            img = perturbation(img)

        elif self.image_transformation == 'rotation':
            img = rotate(img, self.class_to_idx[img_class] * 90, expand=True)


        if self.transform:
            img = self.transform(img.convert('RGB'))


        return img, self.class_to_idx[img_class]
    

class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                # print(name)
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)

        return target_activations, x

def preprocess_image(img):
    # Mean and std for val set
    normalize = transforms.Normalize(mean=[0.4552, 0.4552, 0.4552],
                                    std=[0.2191, 0.2191, 0.2191])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, device=None):
        self.model = model  # pretrained model
        self.feature_module = feature_module  # the feature module of the pretrained model
        self.model.eval()  # set the model to evaluation mode
        self.mps = device  # whether to use GPU
        if self.mps:
            self.model = model.to(device)

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.mps:
            input_img = input_img.to(self.mps)

        self.feature_module.zero_grad()  # set the gradient of feature module to zero
        self.model.zero_grad()  # set the gradient of model to zero

        # output logits and output of the feature module specific layer
        features, output = self.extractor(input_img)  # push the image through the model and extract the features and output

        '''
        In this part you should compute the gradients using the provided tensor 'output'.  
        '''
        ### implement your code here

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)  # 1, 1000
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.mps:
            one_hot = torch.sum(one_hot.to(self.mps) * output)  # multiply by output - zeros for everything except the target category
            # print(one_hot)
        else:
            one_hot = torch.sum(one_hot * output)
            # print(one_hot)

        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        '''
        In this part you should implement Eq. 1
        '''
        # calculate the weights of the feature maps

        target = features[-1]  # 1, 2048, 7, 7
        target = target.cpu().data.numpy()[0, :]  # 2048, 7, 7
        weights = np.mean(grads_val, axis=(2, 3))[0, :]  # 2048
        
        '''
        In this section you should implement Eq. 2
        '''
        # calculate the weighted combination of the feature maps

        cam = np.zeros(target.shape[1:], dtype=np.float32)  # 7, 7
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)  # ReLU

        '''
        In this section, resize the heatmap to the size of input image and normalize it into [0..1]
        '''
        # resize the heatmap to the size of input image
        # normalize it into [0..1]

        cam = cv2.resize(cam, input_img.shape[2:])  # 7, 7 -> 224, 224
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
    
        '''
        Return the resulting heatmap
        '''
        return cam


def run_grad_cam(image_path, grad_cam, target_category, save_path=None):
    '''
    Grad-CAM main function, tweaked version of the one from exercises

            Args:
                image_path (str): path to the image
                grad_cam (GradCam): GradCam object
                target_category (int): target category
                save_path (str): path to save the resulting image
            
            Returns:
                cam (np.array): heatmap

    '''
    img = cv2.imread(image_path, 1) # Read the image with opencv
    img = np.float32(img) / 255

    # Opencv loads as BGR:  BLUE GREEN RED
    img = img[:, :, ::-1]
    input_img = preprocess_image(img)  # Prepare the image for the model - specific for imagenet models


    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = 391
    grayscale_cam = grad_cam(input_img, target_category)


    grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    cam = show_cam_on_image(img, grayscale_cam)

    if save_path is not None:
        cv2.imwrite(save_path, cam)

    return cam


def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.7) -> Image.Image:
    """Overlay a colormapped mask on a background image

    >>> from PIL import Image
    >>> import matplotlib.pyplot as plt
    >>> from torchcam.utils import overlay_mask
    >>> img = ...
    >>> cam = ...
    >>> overlay = overlay_mask(img, cam)

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image

    Raises:
        TypeError: when the arguments have invalid types
        ValueError: when the alpha argument has an incorrect value
    """

    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("img and mask arguments need to be PIL.Image")

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay))[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img

def apply_score_cam(img_tensor, model, layer, normalizer, batch_size, alpha=0.5, device=torch.device('mps')):
    '''
    Apply ScoreCAM to an image tensor
    
        Args:
            img_tensor: image tensor to be processed
            model: model to be used
            layer: layer to be used
            normalizer: normalizer to be used -- to reverse the normalization
            batch_size: batch size to be used
            alpha: alpha value to be used for the overlay
            device: device to be used for the processing

        Returns:
            overlay: overlayed image heatmap

    '''

    cam = ScoreCAM(model, layer, batch_size=batch_size)

    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor.to(device))
    
    class_idx = output.argmax().item()

    cam_map = cam(class_idx)

    img_to_show = transforms.ToPILImage()(normalizer(img_tensor.squeeze(0)))
    cam_to_show = transforms.ToPILImage()(cam_map[0])

    overlay = overlay_mask(img_to_show, cam_to_show, alpha=alpha)
    
    return overlay


def plot_image_grid(images, row_labels, col_labels):
    '''
    Plot a grid of images
    
        Args:
            images: list of images to be plotted
            row_labels: list of labels for each row
            col_labels: list of labels for each column
        
        Returns:
            fig: a matplotlib figure
    '''
    assert len(images) == len(row_labels) * len(col_labels), \
        "The number of images should be equal to len(row_labels) * len(col_labels)"

    fig = plt.figure(figsize=(3*len(col_labels), 3*len(row_labels)))
    gs = gridspec.GridSpec(len(row_labels), len(col_labels))

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax = plt.subplot(gs[i, j])
            img = images[i*len(col_labels)+j]
            ax.imshow(img)
            ax.axis('on')  # Change to 'on' to show separation lines

            # Remove the ticks but keep the grid lines
            ax.set_xticks([])
            ax.set_yticks([])

            # Column labels (only for the first row)
            if i == 0:
                ax.annotate(col_labels[j], xy=(0.5, 1), xytext=(0, 5), 
                            xycoords='axes fraction', textcoords='offset points',
                            ha='center', va='bottom', fontsize=20)

            # Row labels (only for the first column)
            if j == 0:
                ax.annotate(row_labels[i], xy=(0, 0.5), xytext=(-5, 0),
                            xycoords='axes fraction', textcoords='offset points',
                            ha='right', va='center', rotation='vertical', fontsize=20)

    plt.tight_layout()
    plt.show()

    return fig



class RegularizedClassSpecificImageGeneration:
    """
        Produces an image that maximizes a certain class with gradient ascent. Uses Gaussian blur, weight decay, and clipping.
    """

    def __init__(self, model, target_class, device, save_folder_name):
        self.mean = [-0.485, -0.456, -0.406]
        self.std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
        self.model = model.to(device) if use_gpu else model
        self.device = device
        self.model.eval()
        self.target_class = target_class
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
        # Create the folder to export images if not exists
        self.save_folder_name = save_folder_name
        if not os.path.exists(f'./{save_folder_name}/class_{self.target_class}'):
            os.makedirs(f'./{save_folder_name}/class_{self.target_class}')

    def generate(self, iterations=160, blur_freq=4, blur_rad=1, wd=0.0001, clipping_value=0.1):
        """Generates class specific image with enhancements to improve image quality.
        See https://arxiv.org/abs/1506.06579 for details on each argument's effect on output quality.

        Play around with combinations of arguments. Besides the defaults, this combination has produced good images:
        blur_freq=6, blur_rad=0.8, wd = 0.05
        Keyword Arguments:
            iterations {int} -- Total iterations for gradient ascent (default: {150})
            blur_freq {int} -- Frequency of Gaussian blur effect, in iterations (default: {6})
            blur_rad {float} -- Radius for gaussian blur, passed to PIL.ImageFilter.GaussianBlur() (default: {0.8})
            wd {float} -- Weight decay value for Stochastic Gradient Ascent (default: {0.05})
            clipping_value {None or float} -- Value for gradient clipping (default: {0.1})

        Returns:
            np.ndarray -- Final maximally activated class image
        """
        initial_learning_rate = 6
        for i in range(1, iterations):
            # Process image and return variable

            # implement gaussian blurring every ith iteration
            # to improve output
            if i % blur_freq == 0:
                self.processed_image = preprocess_and_blur_image(
                    self.created_image, False, blur_rad, device=self.device)
            else:
                self.processed_image = preprocess_and_blur_image(
                    self.created_image, False, device=self.device)

            if use_gpu:
                self.processed_image = self.processed_image.to(self.device)

            # Define optimizer for the image - use weight decay to add regularization
            # in SGD, wd = 2 * L2 regularization (https://bbabenko.github.io/weight-decay/)
            optimizer = SGD([self.processed_image],
                            lr=initial_learning_rate, weight_decay=wd)
            # Forward
            output = self.model(self.processed_image)
            # Target specific class
            class_loss = -output[0, self.target_class]

            if i in np.linspace(0, iterations, 10, dtype=int):
                print('Iteration:', str(i), 'Loss',
                      "{0:.2f}".format(class_loss.data.cpu().numpy()),
                      'LR:', "{0:.2f}".format(initial_learning_rate))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()

            if clipping_value:
                torch.nn.utils.clip_grad_norm(
                    self.model.parameters(), clipping_value)
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image.cpu())

            if i in np.linspace(0, iterations, 10, dtype=int):
                # Save image
                im_path = f'./{self.save_folder_name}/class_{self.target_class}/c_{self.target_class}_iter_{i}_loss_{class_loss.data.cpu().numpy()}.jpg'
                save_image(self.created_image, im_path)

        # save final image
        im_path = f'./{self.save_folder_name}/class_{self.target_class}/c_{self.target_class}_iter_{i}_loss_{class_loss.data.cpu().numpy()}.jpg'
        save_image(self.created_image, im_path)

        # write file with regularization details
        with open(f'./{self.save_folder_name}/class_{self.target_class}/run_details.txt', 'w') as f:
            f.write(f'Iterations: {iterations}\n')
            f.write(f'Blur freq: {blur_freq}\n')
            f.write(f'Blur radius: {blur_rad}\n')
            f.write(f'Weight decay: {wd}\n')
            f.write(f'Clip value: {clipping_value}\n')

        # rename folder path with regularization details for easy access
        os.rename(f'./{self.save_folder_name}/class_{self.target_class}',
                  f'./{self.save_folder_name}/class_{self.target_class}_blurfreq_{blur_freq}_blurrad_{blur_rad}_wd{wd}')
        return self.processed_image


def preprocess_and_blur_image(pil_im, resize_im=True, blur_rad=None, device=None):
    """
        Processes image with optional Gaussian blur for CNNs
    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
        blur_rad (int): Pixel radius for Gaussian blurring (default = None)
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print(
                "could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im.thumbnail((224, 224))

    # add gaussin blur to image
    if blur_rad:
        pil_im = pil_im.filter(ImageFilter.GaussianBlur(blur_rad))

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    if use_gpu:
        im_as_var = Variable(im_as_ten.to(device), requires_grad=True)
    else:
        im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def pil_image_to_base64(image):
    """Convert a PIL image to a base64 encoded string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def display_images_in_rows(images, max_images_per_row=3, width=100, margin=5):
    """
    Display preloaded images in rows in a Jupyter Notebook.
    
    Parameters:
    images (list): List of PIL.Image objects.
    max_images_per_row (int): Maximum number of images per row.
    width (int): Width of the displayed images.
    margin (int): Margin between images in pixels.
    """
    images_html = []
    for i, img in enumerate(images):
        if i % max_images_per_row == 0:
            images_html.append('<div style="display: flex;">')
        
        images_html.append(
            f'<img style="margin: {margin}px; float: left; border: 1px solid black;" src="data:image/png;base64,{pil_image_to_base64(img)}" width="{width}" />'
        )
        
        if i % max_images_per_row == max_images_per_row - 1 or i == len(images) - 1:
            images_html.append('</div>')
    
    display(HTML(''.join(images_html)))