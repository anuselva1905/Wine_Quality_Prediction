from mlProject.config.configuration import ConfigurationManager
from mlProject.components.model_trainer import ModelTrainer
from mlProject import logger

STAGE_NAME = "Model Trainer stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(f"Starting {STAGE_NAME}")

            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()  # ✅ Get config

            model_trainer = ModelTrainer(config=model_trainer_config)  # ✅ Correct instance
            model_trainer.train()  # ✅ Call train method
            
            logger.info(f"{STAGE_NAME} completed successfully")

        except Exception as e:
            logger.error(f"Error in {STAGE_NAME}: {e}")
            raise e
