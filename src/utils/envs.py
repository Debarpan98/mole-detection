import os

class Envs:
    def __init__(self):
        self.gcs = os.environ["GCS"].lower() in ("true","1")
        self.gcs_image_bucket = os.environ["GCS_IMAGE_BUCKET"]
        self.load_models = os.environ["LOAD_MODELS"].lower() in ("true","1")
        self.user_derminator=os.environ["USER_DERMINATOR"]
        self.pw_derminator = os.environ["PW_DERMINATOR"]
        self.config_file="/home/gauthies/sdd/sdd-general/configs/sdd_config.yaml"
        self.gcs_bucket = "oro-ds-test-bucket"
        
        #"/app/configs/sdd_config.yaml"#"/home/elcinergin/sdd/sdd-general/configs/sdd_config.yaml" # #"/Users/configs/oro/sdd_files/configs/sdd_config.yaml"  #"/app/configs/sdd_config.yaml"  #"/Users/configs/oro/sdd_files/configs/sdd_config.yaml" /home/elcinergin/sdd/sdd_general/sdd_files/sdd_config.yaml
        #"/app/configs/sdd_config.yaml" #"/home/elcinergin/sdd/sdd-general/configs/sdd_config.yaml" # #"/Users/configs/oro/sdd_files/configs/sdd_config.yaml"  #"/app/configs/sdd_config.yaml"  #"/Users/configs/oro/sdd_files/configs/sdd_config.yaml" /home/elcinergin/sdd/sdd_general/sdd_files/sdd_config.yaml
