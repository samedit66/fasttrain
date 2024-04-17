def is_in_notebook():
    
    def is_in_jupyter():
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False      # Probably standard Python interpreter
    
    def is_in_colab():
        try:
            import google.colab
            return True
        except ImportError:
            return False
        
    return is_in_colab() or is_in_jupyter()