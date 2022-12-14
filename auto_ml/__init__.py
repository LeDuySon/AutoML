from .tpot_automl import TpotAutoML
from .pycaret_automl import PycaretAutoML

automl_mapper = {
    "tpot": TpotAutoML,
    "pycaret": PycaretAutoML,
    "auto-sklearn": None
}

def automl_builder(automl_type, configs):
    """Build automl instance

    Args:
        automl_type (str): Type of automl tool that u use  

    Returns:
        BaseAutoML: Instance of your automl tool
    """
    if(automl_type not in automl_mapper):
        raise ValueError(f"{automl_type} is not supported")
    
    automl = automl_mapper[automl_type]
    if(not automl):
        raise ValueError(f"{automl_type} is not implemented")
    
    return automl(configs)