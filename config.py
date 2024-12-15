import yaml
def configuration_yaml():
    with open("./params.yaml", "r") as f:
        return yaml.safe_load(f)
config = configuration_yaml()

