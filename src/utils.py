import inspect
import dis

def UsesArgument(func, argnr):
    """ Returns true if function/lambda uses its argument nr argnr"""
    argname = func.__code__.co_varnames[argnr]
    bytecode = dis.Bytecode(func)
    for instr in bytecode:
        if argname in str(instr):
            return True
    return False

    
