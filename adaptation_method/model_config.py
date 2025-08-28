from INFORM_Project_M.adaptation_method.Autoencoder import ResNetAutoEncoder, Autoencoder
import wandb
def Configuration(model_name = ''):    
    run = wandb.init(
        project= 'Adapter',
        name = f'{model_name}_run',
        config = {
            'batch_size': 32,
            'learning_rate': 1e-3,        
            'epochs': 30,
            'Weight_decay': 1e-4
        }, 
    )
    return run









