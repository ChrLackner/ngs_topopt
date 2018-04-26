import logging

logger = logging.getLogger("ngs_topopt")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

raw_format = logging.Formatter("%(message)s")

ch.setFormatter(raw_format)
logger.addHandler(ch)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

debugfile = logging.FileHandler("ngs_topopt.log")
debugfile.setLevel(logging.DEBUG)
debugfile.setFormatter(formatter)
