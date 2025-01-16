import yaml

class ConfigLoader:
    config_data = None

    @classmethod
    def get_config(cls, file_name, force_load=False):
        if cls.config_data is None or force_load:
            with open(file_name, 'r') as file:
                cls.config_data = yaml.safe_load(file)

        return cls.config_data