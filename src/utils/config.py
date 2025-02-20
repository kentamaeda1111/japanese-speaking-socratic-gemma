import os
from typing import Dict
from dotenv import load_dotenv
from IPython.display import display
from ipywidgets import widgets

def get_colab_api_keys() -> Dict[str, str]:
    """Get API keys using Colab forms"""
    try:
        from google.colab import output
        print("Running in Colab environment. Please enter your API keys:")
        
        api_keys = {}
        form = widgets.VBox([
            widgets.Password(description='CLAUDE_API_KEY_1:', layout=widgets.Layout(width='500px')),
            widgets.Password(description='CLAUDE_API_KEY_2:', layout=widgets.Layout(width='500px')),
            widgets.Password(description='CLAUDE_API_KEY_QUALITY:', layout=widgets.Layout(width='500px')),
            widgets.Password(description='CLAUDE_API_KEY_QUALITY2:', layout=widgets.Layout(width='500px')),
            widgets.Password(description='HUGGINGFACE_API_KEY:', layout=widgets.Layout(width='500px'))
        ])
        display(form)
        
        # Wait for user input
        output.eval_js('new Promise(resolve => setTimeout(resolve, 1000))')
        
        api_keys = {
            'claude_api_key_1': form.children[0].value,
            'claude_api_key_2': form.children[1].value,
            'claude_api_key_quality': form.children[2].value,
            'claude_api_key_quality2': form.children[3].value,
            'huggingface_api_key': form.children[4].value
        }
        
        # Set environment variables
        for env_var, value in {
            'CLAUDE_API_KEY_1': api_keys['claude_api_key_1'],
            'CLAUDE_API_KEY_2': api_keys['claude_api_key_2'],
            'CLAUDE_API_KEY_QUALITY': api_keys['claude_api_key_quality'],
            'CLAUDE_API_KEY_QUALITY2': api_keys['claude_api_key_quality2'],
            'HUGGINGFACE_API_KEY': api_keys['huggingface_api_key']
        }.items():
            os.environ[env_var] = value
            
        return api_keys
    except ImportError:
        return None

def get_api_keys() -> Dict[str, str]:
    """
    Load API keys from environment variables, Kaggle secrets, or Colab forms.
    
    Returns:
        Dict[str, str]: Dictionary containing API keys
        {
            'claude_api_key_1': str,
            'claude_api_key_2': str,
            'claude_api_key_quality': str,
            'claude_api_key_quality2': str,
            'huggingface_api_key': str
        }
    
    Raises:
        ValueError: If required API keys are not found
    """
    # Try to load from .env file first (for local environment)
    load_dotenv(override=True)
    
    required_keys = {
        'claude_api_key_1': 'CLAUDE_API_KEY_1',
        'claude_api_key_2': 'CLAUDE_API_KEY_2',
        'claude_api_key_quality': 'CLAUDE_API_KEY_QUALITY',
        'claude_api_key_quality2': 'CLAUDE_API_KEY_QUALITY2',
        'huggingface_api_key': 'HUGGINGFACE_API_KEY'
    }
    
    # Get API keys from environment variables
    api_keys = {}
    missing_keys = []
    
    for key, env_var in required_keys.items():
        value = os.getenv(env_var)
        if value is None:
            missing_keys.append(env_var)
        api_keys[key] = value
    
    # If keys are missing, try different methods based on environment
    if missing_keys:
        # Try Kaggle secrets
        if os.path.exists('/kaggle/working'):
            try:
                from kaggle_secrets import UserSecretsClient
                user_secrets = UserSecretsClient()
                
                for env_var in missing_keys[:]:
                    try:
                        secret_value = user_secrets.get_secret(env_var)
                        os.environ[env_var] = secret_value
                        missing_keys.remove(env_var)
                    except Exception as e:
                        print(f"Could not get {env_var} from Kaggle secrets: {str(e)}")
                
                # Update api_keys with new values
                for key, env_var in required_keys.items():
                    value = os.getenv(env_var)
                    if value is not None:
                        api_keys[key] = value
                        
            except ImportError:
                print("Not running in Kaggle environment or secrets not accessible")
        
        # Try Colab forms
        elif os.path.exists('/content'):
            colab_keys = get_colab_api_keys()
            if colab_keys:
                api_keys.update(colab_keys)
                missing_keys = []
    
    # If still missing keys, raise error
    if missing_keys:
        raise ValueError(
            f"Missing required API keys: {', '.join(missing_keys)}\n"
            "Please either:\n"
            "1. Set them in your .env file (for local environment)\n"
            "2. Set them in Kaggle secrets (for Kaggle environment)\n"
            "3. Enter them in the form (for Colab environment)"
        )
    
    return api_keys 