import logging

class MatplotlibLogger:
    def __init__(self, filename=None):
    
        if not filename: filename = 'results'
        self.logfilename = filename
        # Create the logger object
        self.logger = logging.getLogger('TuningData')
        self.logger.setLevel(logging.DEBUG)
    
        # Create console handler and set level to info
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        
        # Custom formatter
        formatter = logging.Formatter('#%(asctime)s\n%(message)s')
        ch.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(ch)
    
        # File name
        self.logger.addHandler(logging.FileHandler(filename))

    def getLogger(self):
        return self.logger



   
