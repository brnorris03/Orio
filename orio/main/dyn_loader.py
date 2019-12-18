#
# Dynamic loader is used to dynamically load a Python module and a Python class
#

import importlib
from .util.globals import *


# ----------------------------------------------

class DynLoader:

    def __init__(self):
        '''To instantiate a dynamic loader object'''

        self.__loaded_modules = {}
        self.__loaded_classes = {}

    # -------------------------------------------------

    def __loadModule(self, mod_name):
        '''To dynamically load the specified module'''

        # check if the module has already been loaded before
        key_name = mod_name
        if key_name in self.__loaded_modules:
            return self.__loaded_modules[key_name]

        # load the specified module dynamically
        try:
            topmodule = importlib.import_module(mod_name)
        except Exception as e:
            err('orio.main.dyn_loader: Failed to load module "%s"\n --> %s: %s' % (mod_name, e.__class__.__name__, e))

        # return the loaded module
        return topmodule

    # -------------------------------------------------

    def loadClass(self, mod_name, class_name):
        '''To dynamically load the specified class from the specified module'''

        # check if the class has already been loaded before
        key_name = '%s %s' % (mod_name, class_name)
        if key_name in self.__loaded_classes:
            return self.__loaded_classes[key_name]

        # load the specified module dynamically
        module = self.__loadModule(mod_name)

        debug('orio.main.dyn_loader: successfully loaded module %s' % module)
        # load the specified class from the loaded module
        try:
            mod_class = getattr(module, class_name)
        except:
            err('orio.main.dyn_loader:  no class "%s" defined in "%s"' % (class_name, mod_name))

        # remember the currently loaded class
        self.__loaded_classes[key_name] = mod_class

        # return the loaded class
        return mod_class
