# Deep learning project

The aim of this academic project is to develop and evaluate the predictive capabilities of few distinct 
models, leveraging the principles of fully supervised and self-supervised learning techniques. The first 
two models are based on fully supervised learning, where the task is to classify different scenes from 
static images. These models will be trained using images paired with corresponding scene annotations.  
However, fully supervised learning is often limited by the need for extensive annotated data, which is 
expensive to acquire. To overcome this limitation, self-supervised training methods are proposed. These 
methods employ a technique called a 'pretext task'—a secondary process that helps extract generic 
feature representations from unannotated data. The objective is to adapt this feature representation for 
other tasks or use it as is. Following this, the aim is to compare these models. This involves determining 
their prediction accuracy, scrutinizing the visual explanations for the predictions made by each model, 
and interpreting the features internally encoded by the models visually.

# Files information

- ANN_project.pdf - task description
- report_JakubnNiedziela.pdf - written pdf report
- explore_dataset.ipynb - basic info on the data used
- supervised_model_v{1,2}.ipynb - notebooks for training the model on 15SceneData dataset - adjusting a pretrained efficientnet (all model weights free to change)
- supervised_pretrained.ipynb - notebook for adjusting the pretrained model (only the classifier part, CNN part stays the same)
- semi_supervised_rotation.ipynb - notebook for training the model on dataset with augmented images (rotations by 90, 180 and 270 degrees)
- rotation_classification_v1.ipynb - notebook for final training of rotation model on 15SceneData
- semi_supervised_perturbation.ipynb - notebook for training the model on dataset with augmented images (adding black or white square into the image)
- images_perturbation.ipynb - notebook with visualisations of different image perturbations
- perturbation_classification_v1.ipynb - notebook for final training of perturbation model on 15SceneData
- score_cam.ipynb - notebook for running and plotting score cam method (more info in: https://arxiv.org/abs/1910.01279)
- inversion.ipynb - notebook for running and plotting inversion explainatory method (more info in: https://arxiv.org/abs/1412.0035)
- helpers.py - functions used for training, loading data, visualisations
- misc_function.py - functions for inversion (https://github.com/alexstoken/pytorch-cnn-visualizations/blob/master/src/misc_functions.py)

# Folders info:
- 15SceneData - dataset used
- models - saved trained models
- plots - plots with training, inversion and score cam results