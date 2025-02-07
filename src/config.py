import logging
import os
import yaml

from enum import Enum

class ConfigType(Enum):
    YAML = ".yaml"
    NONE = None

class Configuration():
    def __init__(self, 
                 directory, 
                 default_name, 
                 config_type=ConfigType.YAML, 
                 logger=None):
        """Constructor for the main configuration class.

        Args:
            directory (Path):   Path to the configuration directory
            default_name (str): Name of the configuration file
            config_type (ConfigType, optional): Filetype for the configuration 
                file. Defaults to ConfigType.YAML.
            logger (logger | None, optional): Logging object, if none is 
                specified the normal python logger is used. Defaults to None.
        """   
        self.name           = default_name
        self.directory      = directory
        self.__logger       = logger if logger is not None else logging
        self.__loaded       = False
        self.sections       = {}
        self.section_paths  = {}
        
        match config_type:
            case ConfigType.YAML:
                self.extension = ConfigType.YAML.value
            case _:
                self.__logger.error(
                    f"Config type {config_type.value} is not supported.") 
                return
            

    def __getitem__(self, section_name):
        if not self.__loaded:
            self.__logger.error(
                "Trying to access Section before config was loaded")
            return _InvalidSection(self.__logger)
        
        if section_name not in self.sections:
            self.__logger.error(f"{section_name} is not a defined section.") 
            return _InvalidSection(self.__logger)
        
        return self.sections[section_name]
                
    def __str__(self):
        string = "\n"

        for key, value in self.sections.items():
            string += key + "\n"
            string += str(value)

        return string
    
    def __prepare_sections(self, sections, config_path):
        for section_name, section in sections.items(): 
            if "external" in section:
                self.__add_external_section(section_name, section)
            else:
                self.__add_section(section_name, 
                                   config_path,
                                   section)

    def __get_external_section(section):
        external_path       = os.path.join(*section.split("/"))
        external_section    = Configuration.__load_config_file(external_path)
        
        if not external_section:
            return None
        
        return external_section, external_path

    def __add_external_section(self, section_name, section):
            external_section, external_path = (
                Configuration.__get_external_section(section))
            
            if (external_section 
                and "only_load_selection" in section 
                and "selection" in section
                and section["selection"] in external_section):

                external_section = external_section[section["selection"]]

                self.__add_section(section_name, 
                                   external_path, 
                                   external_section,
                                   read_only=True)
                return
            
            #self.__add_section(section_name, external_path, external_sections)
            
    def __add_section(self, 
                      section_name, 
                      section_path, 
                      section, 
                      read_only = False):

        self.sections[section_name] = Section(section_name, 
                                              section,
                                              logger = self.__logger, 
                                              read_only = read_only)
        
        self.add_section_path(section_path, section_name)

    def __load_config_file(config_path):
        loaded_data = {}
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                loaded_data = yaml.safe_load(file)
        
        return loaded_data
            
    def add_section_path(self, section_path, section_name):
        self.section_paths.setdefault(section_path, []).append(section_name) 

    def get_load_options(self):
        files = []

        for file in os.listdir(self.directory):
            if (os.path.isfile(os.path.join(self.directory, file)) 
                and file.endswith(self.extension)):
                
                files.append(file)
        
        return files

    def load_defaults(self):
        self.load_configuration(self.name)

    def load_configuration(self, config_file = None):
        """Loads configuration data from the specified file."""
        if config_file is None:
            config_path = os.path.join(self.directory, 
                                       self.name + self.extension)            
        else:
            config_path = os.path.join(self.directory, 
                                       config_file + self.extension)
        
        loaded_data = Configuration.__load_config_file(config_path)

        if loaded_data:
            self.__prepare_sections(loaded_data, config_path)
            self.__loaded = True
            return True
        elif config_file is not self.name:
            self.__logger.warning(
                f"No config file found at {config_path}, loading defaults.")
            self.load_defaults()
        else:
            self.__logger.error("Default file could not be loaded.")
            return False
     
    def load_usages(self):
        usages = self.__load_config_file("todo")

        if not usages:
            return 

    # distribute usage to items
        
    # def save(self, 
    #          directory=constants.CONFIG_PATH, 
    #          config_file=constants.CONFIG_DEFAULT_FILE):
        
    #     """Saves the current configuration data to the specified file."""
    #     config_path = os.path.join(directory, config_file)

    #     with open(config_path, 'w') as file:
    #         yaml.dump(self.config_data, file, indent=4)   




class Section():
    def __init__(self, 
                 section_name, 
                 section,
                 logger,
                 read_only = False):
        
        self.name           = section_name
        self.description    = "Description not set yet"
        self.__logger       = logger
        self.read_only      = read_only

        self.configurations = self.__prepare_items(section)

    def __getitem__(self, configuration_name):
        if configuration_name in self.configurations:
            if isinstance(self.configurations[configuration_name], Section):
                return self.configurations[configuration_name]
            return self.configurations[configuration_name].value
        else:
            self.__logger.error(f"{configuration_name} is not a valid item") 
        return None
    
    def __setitem__(self, configuration_name, configuration):
        if self.read_only:
            self.__logger.error(f"{self.name} is read-only") 
            return

        if configuration_name in self.configurations:
            self.configurations[configuration_name].value = configuration
        else:
            self.__logger.error(f"{configuration_name} is not a valid item") 

    def __contains__(self, configuration_name):
        return configuration_name in self.configurations
    
    def __str__(self, level=1):
        string = ""

        for key, value in self.configurations.items():
            string += "\t" * level
            string += key + ":\n"
            string += value.__str__(level + 1)

        return string
    
    def __prepare_items(self, section):
        configurations = {}
        for item_name, item in section.items():
            if not isinstance(item, dict):
                configurations[item_name] = Item(item)
            elif "value" in item:
                configurations[item_name] = Item(item["value"])
            elif "external" in item:
                external_section, _ = (
                    Configuration.get_external_section(item))
                if not external_section:
                    continue
                configurations[item_name] = Section(item_name, 
                                                    external_section[
                                                        item["selection"]], 
                                                    self.__logger)
            else:
                configurations[item_name] = Section(item_name, 
                                                    item, 
                                                    self.__logger)

        return configurations

    def get_usage(self, configuration_name):
        if configuration_name in self.configurations:
            return self.configurations[configuration_name].usage
        return ""
    
    def get_description(self):
        return self.description
        
    def items(self):
        return self.configurations.items()
    
    def values(self):
        values = []
        for name, config in self.configurations.items():
            values.append((name, config.value))
        return values


class Item():
    def __init__(self, value):
        self.value  = value
        self.usage  = "No usage set yet"
        self.type   = type(value) 
        
        self.__is_list = issubclass(list, self.type)

    def __str__(self, level=2):
        return "\t" * level + str(self.value) + "\n"

    def __contains__(self, entry):
        if not self.__is_list:
            return False
        return entry in self.value
    

    def set_usage(self, usage):
        self.usage  = usage   



class _InvalidSection(Section):
    _singleton = None
    
    def __new__(cls, *args, **kwargs):
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton
    
    def __init__(self, logger):
        self.name           = "Invalid Section"
        self.description    = "Invalid Section, add this section to the config"
        self.configurations = {}
        self.__logger       = logger

    def __getitem__(self, configuration_name):
        self.__logger.error(f"Accesssing item in an invalid section.") 
        