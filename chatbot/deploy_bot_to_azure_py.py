import os
import json
from pathlib import Path

from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice

root_config_path = os.path.join(Path(__file__).resolve().parents[1], 'azure_config.json')

with open(root_config_path, 'r') as f:
    cfg = json.load(f)

subscription_id = cfg["subscription_id"]
resource_group = cfg["resource_group"]
workspace_name = cfg["workspace_name"]
region = cfg["region"]
redeploy_if_exists = bool(cfg.get("redeploy_if_exists", True))

bot_cfg = cfg["services"]["chatbot"]
model_path = bot_cfg["model_path"]
model_name = bot_cfg["model_name"]
service_name = bot_cfg["service_name"]
entry_script = bot_cfg["entry_script"]
pip_packages = bot_cfg.get("pip_packages", ["azureml-defaults"]) 

cpu_cores = int(cfg.get("aci_defaults", {}).get("cpu_cores", 1))
memory_gb = int(cfg.get("aci_defaults", {}).get("memory_gb", 1))

# Get or create workspace
try:
    ws = Workspace.get(name=workspace_name,
                       subscription_id=subscription_id,
                       resource_group=resource_group)
except Exception:
    ws = Workspace.create(name=workspace_name,
                          subscription_id=subscription_id,
                          resource_group=resource_group,
                          location=region,
                          exist_ok=True)

print(f"Using workspace: {ws.name}")

# Ensure model path exists
model_path_abs = os.path.abspath(model_path)
if not os.path.exists(model_path_abs):
    os.makedirs(model_path_abs, exist_ok=True)
    placeholder = os.path.join(model_path_abs, 'placeholder.txt')
    if not os.path.exists(placeholder):
        with open(placeholder, 'w') as ph:
            ph.write('placeholder model artifact')

# Register model
registered_model = Model.register(model_path=model_path_abs, model_name=model_name, workspace=ws)
print(f"Registered model: {registered_model.name}:{registered_model.version}")

# Environment
env = Environment('wealtharena-bot-env')
env.python.conda_dependencies = CondaDependencies.create(pip_packages=pip_packages)

# Inference config
inference_config = InferenceConfig(entry_script=entry_script, environment=env)

# ACI config
aci_config = AciWebservice.deploy_configuration(cpu_cores=cpu_cores, memory_gb=memory_gb)

# If service exists
service = None
try:
    service = Webservice(name=service_name, workspace=ws)
    if redeploy_if_exists:
        print(f"Deleting existing service: {service_name}")
        service.delete()
        service = None
    else:
        print(f"Service {service_name} already exists. Skipping deployment.")
except Exception:
    service = None

if service is None:
    service = Model.deploy(workspace=ws,
                           name=service_name,
                           models=[registered_model],
                           inference_config=inference_config,
                           deployment_config=aci_config)
    service.wait_for_deployment(show_output=True)

print(f"Scoring URI: {service.scoring_uri}")

