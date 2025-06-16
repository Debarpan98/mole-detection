from src.data.augmentation_factory import augmentationFactory
from src.utils.utils import get_mlflow_exp_name

class TrainingHyperparameters():
    def __init__(self, model_name: str, params: dict):
        """_summary_

        Args:
            model_name (str): model name
            params (dict): dictionary of hyperparameters (see parameters below)
                learning_rate (float, optional): learning rate. Defaults to 0.001.
                momentum (float, optional): momentum. Defaults to 0.9.
                gamma (float, optional): gamma_. Defaults to 0.1.
                scheduler_steps (int, optional): scheduler_steps. Defaults to 2.
                optimizer_name (str, optional): optimizer name. Defaults to 'sgd'.
                scheduler_name (str, optional): sceduler name. Defaults to 'steplr'.
                weight_decay (float, optional): weight decay. Defaults to 0.0005.
                loss_function (str, optional): loss fucntion name. Defaults to 'ce'.
                freeze_layers (bool, optional): indicating if some layers are frpzen. Defaults to True.
                percentage_freeze (int, optional): percentage layers or blocks to froze. Defaults to 4.
                epochs (int, optional): number of epochs. Defaults to 10.
                gpus (list, optional): gpus. Defaults to [0,1].
        """
        self.params_dict = params
        self.model_name = model_name 
        self.learning_rate = params['learning_rate']
        self.momentum=params['momentum']
        self.gamma = params['gamma']            
        self.optimizer_name = params['optimizer']
        self.scheduler_name = params['scheduler']
        self.scheduler_steps = params['scheduler_steps']    
        self.weight_decay= params['weight_decay']
        self.loss_function = params['loss']
        self.freeze_layers = params['freeze_layers']
        self.percentage_freeze  = params['percentage_freeze']
        self.epochs = params['epochs']
        self.gpus = params['gpus']
        self.augmentation = params['augmentation']
        self.batch_size = params['batch_size']
        self.gradient_acc_batch = params['gradient_acc_batch']
        self.gradient_acc = params['gradient_acc']
        self.size = params['size']
        self.num_workers = int(params['num_workers'])
        self.visible_gpus = str(params["visible_gpus"])
        self.multi_label_threshold= params['multi_label_threshold']
        self.accum_step_multiple = get_accum_step_multiple(self.gradient_acc, self.batch_size, self.gradient_acc_batch)
        self.focal_gamma = params['focal_gamma']
        self.focal_alpha = params['focal_alpha']
        self.skin_ratio_threshold= params['skin_ratio_threshold']
        self.resized_crop_scale = params['resized_crop_scale']
        self.onecycle_three_phase= params['onecycle_three_phase']
        self.save_model = params['save_model']
        self.process_num = params['process']
        self.image_size=params['image_size'] # original image size (before cropping)'
        self.test_set_from_file = params['test_set_from_file']
        self.test_set_path = params['test_set_path']
        self.val_set_from_file = params['val_set_from_file']
        self.val_set_path = params['val_set_path']
        self.train_set_from_file = params['train_set_from_file']
        self.train_set_path = params['train_set_path']
        self.check_ratio= params['check_ratio']
        self.subsamble_set = params['subsamble_set']
        self.ml_decoder= params['ml_decoder']
        self.early_stopping = params['early_stopping']
        self.val_set = params['val_set']
        self.patience = params['patience']
        self.asymmetric_gamma_neg = params['asymmetric_gamma_neg']
        self.asymmetric_gamma_pos = params['asymmetric_gamma_pos']
        self.merge_acne = params['merge_acne']
        self.asymmetric_clip = params['asymmetric_clip']
        self.mask_skin= params['masking_skin']
        self.one_tag = params['one_tag'] # using images with only one label 
        self.task=params['task']  #['multi_label', 'single_label', 'binary']
        self.warmup_epochs=params['warmup_epochs']
        self.cosine_cycle_epochs = params['cosine_cycle_epochs']
        self.cosine_cycle_decay = params['cosine_cycle_decay']
        self.mixup_alpha =params['mixup_alpha']
        self.mixup = params['mixup']
        
        self.set_task_variables()
        self.verify_val_set()

        if self.process_num is None: 
            self.str_name = self.model_name 
        else:
            self.str_name = self.model_name + '-' + str(self.process_num)

    def verify_val_set(self):
        if  self.early_stopping and not self.val_set:
            self.val_set = True

    def set_test_percentage(self,test_percentag):
        self.test_percentag = test_percentag

    def set_random_state(self, random_state):
        self.random_state = random_state
    
    def set_classes(self, classes):
        self.classes = classes

    def set_skin_threshold(self, threshold):
        self.skin_threshold =threshold
    
    def set_task_variables(self):
        if self.task == 'multi_label':
            self.multi_label = True
            self.binary_classification = False
        elif self.task=='binary':
            self.classes = ['acne', 'not_acne']
            self.multi_label = False
            self.binary_classification = True
        elif self.task=='single_label':
            self.multi_label = False
            self.binary_classification = False
        else:  
            raise NotImplemented(f"Task {self.task} not implemented. The implemented tasks are: multi_label, single_label and binary.")

    def set_samples_values(self, subsample_max_values, subsample_max_per_disease):
        self.subsample_max_values = subsample_max_values
        self.subsample_max_per_disease = subsample_max_per_disease
    
    def set_samples_values(self, subsample_max_values, subsample_max_per_disease):
        self.subsample_max_values = subsample_max_values
        self.subsample_max_per_disease = subsample_max_per_disease

    def set_mlflow_exp_name(self, mlflow_exp_name):
        """Set the mlflow experiment name based on the task
        """    
        try:
            self.mlflow_exp_name = get_mlflow_exp_name(mlflow_exp_name, self.binary_classification, self.multi_label)   
        except: 
            raise Exception('The variables called: binary_classification and multi_label have not been defined. Please make sure these variables are defined before calling set_mlflow_exp_name. ')

    # def set_multi_label(self, multi_label):
    #     self.multi_label = multi_label
    #     if multi_label:
    #             self.description = 'Multi label Classification'
    #     # if self.multi_label and self.loss_function!='bce':
    #     #     raise Exception('The task is a multi-label task. However, you are not using a loss function designed for it.')
    #     if not self.multi_label and self.loss_function=='bce':
    #          raise Exception('The task is a single-label task. However, you are not using a loss function designed for it.')

    # def set_is_binary_classification(self, is_binary_classification):
    #     if is_binary_classification:
    #         self.classes = ['acne', 'not_acne']
    #         self.description = 'Binary Classification'
    #     self.binary_classification = is_binary_classification

def get_accum_step_multiple(gradient_acc, batch_size, gradient_acc_batch ):
        
    if gradient_acc:
        if gradient_acc_batch % batch_size != 0:
             raise Exception('ERROR: Gradient_acc_batch must be a mutliple of batch_size.')
        accum_step_multiple = gradient_acc_batch /  batch_size
    else:
        accum_step_multiple=1
    return accum_step_multiple

def set_training_settings(config, params):
    params.set_random_state(config.get("random_state"))
    params.set_test_percentage(config.get("test_percentage"))
    params.set_classes(config.get("class_list"))
    #params.set_multi_label(config.get("multi_label"))
    params.set_skin_threshold(config.get("skin_ratio_threshold"))
    #params.set_is_binary_classification(config.get("binary_classification"))
    params.set_samples_values(config.get("subsample_max_values"), config.get("subsample_max_per_disease"))
    params.gcs_bucket = config.get("gcs_bucket")

def generate_transforms(params):
    transform= {}
    transform['train'] = augmentationFactory(params.augmentation, params.size, resized_crop_scale= params.resized_crop_scale, image_scale=params.image_size)
    transform['val'] = augmentationFactory('noaugment', params.size, image_scale=params.image_size)
    transform['test'] = augmentationFactory('noaugment', params.size, image_scale=params.image_size)
    return transform

