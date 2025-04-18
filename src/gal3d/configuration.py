import logging

#from .gal3d_main import Galaxy3d

# from https://yellowduck.be/posts/coloring-python-logging-output/




from .util.string_format import string_formator

class ColorFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""
    
    FORMATS = {
        logging.DEBUG: "".join([string_formator("[%(asctime)s.%(msecs)03d]",italics=True,),
                                string_formator(" <%(filename)s>",fg_color='bright_blue',underline=True),
                                string_formator(" line: %(lineno)d ",fg_color='purple',italics=True),"\n", "  >>>  ",
                                string_formator("| %(levelname)s | ",fg_color='cyan',bold=True),"%(message)s"]),
        
        
        logging.INFO: "".join([string_formator("[%(asctime)s.%(msecs)03d]",italics=True,underline=False),
                               string_formator(" <%(filename)s>", fg_color='bright_blue',underline=True),
                               "\n", "  >>>  ",
                                string_formator("| %(levelname)s | ",fg_color='green',bold=True),"%(message)s"]),
        
        logging.WARNING: "".join([string_formator("[%(asctime)s.%(msecs)03d]",fg_color='yellow',italics=True,underline=False),
                                string_formator(" <%(filename)s>",fg_color='bright_blue',underline=True),
                                string_formator(" line: %(lineno)d ",fg_color='purple',italics=True),"\n", "  >>>  ",
                                string_formator("| %(levelname)s | ",fg_color='yellow',bold=True),"%(message)s"]),
        
        logging.ERROR: "".join([string_formator("[%(asctime)s.%(msecs)03d]",fg_color='red',italics=True,underline=False),
                                string_formator(" <%(filename)s>",fg_color='bright_blue',underline=True),
                                string_formator(" line: %(lineno)d ",fg_color='purple',italics=True),"\n", "  >>>  ",
                                string_formator("| %(levelname)s | %(message)s",fg_color='red',bold=True)]),
        
        logging.CRITICAL: "".join([string_formator("[%(asctime)s.%(msecs)03d]",fg_color='red',italics=True,underline=False),
                                string_formator(" <%(filename)s>",fg_color='bright_blue',underline=True),
                                string_formator(" line: %(lineno)d ",fg_color='purple',italics=True),"\n", "  >>>  ",
                                string_formator("| %(levelname)s | %(message)s",fg_color='red',bg_color='white',bold=True)]),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt,datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)
    
class NoColorFormatter(logging.Formatter):
    import re
    FORMATS = {
        logging.DEBUG:"[%(asctime)s.%(msecs)03d] <%(filename)s>  line: %(lineno)d \n   >>>  | %(levelname)s | %(message)s",
        logging.INFO:"[%(asctime)s.%(msecs)03d] <%(filename)s>  line: %(lineno)d \n   >>>  | %(levelname)s | %(message)s",
        logging.WARNING:"[%(asctime)s.%(msecs)03d] <%(filename)s>  line: %(lineno)d \n   >>>  | %(levelname)s | %(message)s",
        logging.ERROR:"[%(asctime)s.%(msecs)03d] <%(filename)s>  line: %(lineno)d \n   >>>  | %(levelname)s | %(message)s",
        logging.CRITICAL:"[%(asctime)s.%(msecs)03d] <%(filename)s>  line: %(lineno)d \n   >>>  | %(levelname)s | %(message)s",}
    ANSI_ESCAPE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    def format(self,record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        if record.levelno ==25 :
            return self.ANSI_ESCAPE.sub('',formatter.format(record))
        return formatter.format(record)
    
def _setup_logging():
    logger = logging.getLogger('gal3d')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    ch.setFormatter(ColorFormatter())
    logger.addHandler(ch)
    
    fh = logging.FileHandler("./gal3d.log", mode='w',  encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(NoColorFormatter())
    logger.addHandler(fh)
    return logger



def set_logging_level(level = logging.INFO):
    """
    Set to logging.INFO for more verbose output, or logging.WARNING for less.
    """
    logger = logging.getLogger('gal3d')
    logger.setLevel(level)
    
logger = _setup_logging()

logo=f"""
                                         .--,-``-.                    
    ,----..                    ,--,     /   /     '.       ,---,      
   /   /   \                 ,--.'|    / ../        ;    .'  .' `\    
  |   :     :                |  | :    \ ``\  .`-    ' ,---.'     \   
  .   |  ;. /                :  : '     \___\/   \   : |   |  .`\  |  
  .   ; /--`      ,--.--.    |  ' |          \   :   | :   : |  '  |  
  ;   | ;  __    /       \   '  | |          /  /   /  |   ' '  ;  :  
  |   : |.' .'  .--.  .-. |  |  | :          \  \   \  '   | ;  .  |  
  .   | '_.' :   \__\/: . .  '  : |__    ___ /   :   | |   | :  |  '  
  '   ; : \  |   ," .--.; |  |  | '.'|  /   /\   /   : '   : | /  ;   
  '   | '/  .'  /  /  ,.  |  ;  :    ; / ,,/  ',-    . |   | '` ,/    
  |   :    /   ;  :   .'   \ |  ,   /  \ ''\        ;  ;   :  .'      
   \   \ .'    |  ,     .-./  ---`-'    \   \     .'   |   ,.'        
    `---`       `--`---'                 `--`-,,-'     '---'          """
logo_color = "\n".join([string_formator(logo.split('\n')[i],bg_color=(0+5*i,0+5*i,0+5*i),fg_color='white',bold=True,italics=True) for i in range(len(logo.split('\n')))])